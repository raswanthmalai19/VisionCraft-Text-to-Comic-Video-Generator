import torch
import os
import base64
from io import BytesIO
import re
import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Handle potential dependency conflicts gracefully
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers loaded successfully")
except Exception as e:
    DIFFUSERS_AVAILABLE = False
    print(f"⚠️ Diffusers import failed: {e}")
    print("⚠️ Consider updating packages: pip install --upgrade torch torchvision transformers diffusers")
    
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available")

try:
    import textwrap
    TEXTWRAP_AVAILABLE = True
except ImportError:
    TEXTWRAP_AVAILABLE = False

# Advanced NLP and ML imports
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.tree import Tree
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available, using basic text processing")

try:
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModel,
        T5Tokenizer,
        T5ForConditionalGeneration
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
    print("✅ Advanced NLP models available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available, using fallback processing")

# -------------------------
# CONFIG
# -------------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "output"
PANELS_PER_ROW = 2
IMAGE_SIZE = (512, 512)
FONT_SIZE = 20
MARGIN = 20

# Dynamic panel configuration
MIN_PANELS = 2
MAX_PANELS = 20  # Increased limit since we support multiple pages
OPTIMAL_WORDS_PER_PANEL = 15

# Multi-page configuration
MIN_PANELS_PER_PAGE = 4
MAX_PANELS_PER_PAGE = 8
PREFERRED_PANELS_PER_PAGE = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# COMIC THEMES
# -------------------------
COMIC_THEMES = {
    'classic': {
        'name': 'Classic Comic Book',
        'style': 'classic American comic book style, clean line art, vibrant primary colors, professional comic illustration, superhero comic aesthetic, detailed character design'
    },
    'manga': {
        'name': 'Manga/Anime Style', 
        'style': 'Japanese manga style, detailed line art, expressive large eyes, dynamic action poses, black and white with screentone effects, anime character design'
    },
    'superhero': {
        'name': 'Superhero Comic',
        'style': 'superhero comic book style, dynamic action scenes, muscular heroic characters, bold primary colors, dramatic lighting, Marvel DC comic aesthetic'
    },
    'cartoon': {
        'name': 'Cartoon Style',
        'style': 'animated cartoon style, simple clean lines, bright cheerful colors, friendly rounded character designs, Disney-like animation aesthetic'
    },
    'noir': {
        'name': 'Film Noir',
        'style': 'film noir comic style, black and white artwork, dramatic high contrast shadows, vintage 1940s detective aesthetic, moody atmospheric lighting'
    },
    'fantasy': {
        'name': 'Fantasy Adventure',
        'style': 'fantasy comic book style, magical elements, medieval setting, detailed armor and weapons, mystical atmosphere, epic fantasy illustration'
    },
    'scifi': {
        'name': 'Science Fiction',
        'style': 'science fiction comic style, futuristic technology, space settings, cyberpunk elements, metallic colors, advanced sci-fi equipment'
    },
    'webcomic': {
        'name': 'Modern Web Comic',
        'style': 'modern digital webcomic style, clean vector art, contemporary character design, simplified but expressive art style'
    }
}

# -------------------------
# FONT
# -------------------------
try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE)
except:
    font = ImageFont.load_default()

# -------------------------
# LOAD MODELS (lazy loading)
# -------------------------
pipe = None
text_encoder = None
sentence_model = None
summarization_model = None

# Initialize NLTK (try to download required data)
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        except:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            except:
                print("⚠️ Could not download NLTK data, using basic text splitting")
                NLTK_AVAILABLE = False

def load_models():
    """Load all required models (called once on first use)."""
    global pipe, text_encoder, sentence_model, summarization_model
    
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("Diffusers not available. Please fix dependency conflicts and restart.")
    
    # Load Stable Diffusion model
    if pipe is None:
        print("Loading Stable Diffusion model...")
        
        try:
            # Check if model exists in system cache
            default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--runwayml--stable-diffusion-v1-5")
            if os.path.exists(default_cache):
                print("✅ Found existing model in system cache, loading...")
            else:
                print("⬇️ Downloading model for the first time...")
            
            # Determine device - prefer CUDA if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(device)
            
            # Apply speed optimizations
            if device == "cuda":
                print("🚀 Applying GPU optimizations...")
                # Enable memory efficient attention if available
                try:
                    pipe.enable_attention_slicing(1)
                    print("   ✅ Attention slicing enabled")
                except:
                    pass
                
                # Enable VAE slicing for lower memory usage
                try:
                    pipe.enable_vae_slicing()
                    print("   ✅ VAE slicing enabled")
                except:
                    pass
                
                # Skip torch.compile - it takes too long on first run
                # and RTX 3050 Ti has limited VRAM
                print("   ✅ Ready for inference (torch.compile disabled for faster startup)")
            else:
                print("⚠️ Running on CPU - generation will be slower")
                print("   💡 Tip: Install PyTorch with CUDA for 10-50x faster generation")
            
            print("✅ Stable Diffusion model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load Stable Diffusion model: {e}")
            raise
        except Exception as e:
            print(f"❌ Failed to load Stable Diffusion model: {e}")
            raise
    
    # Load NLP models if available
    if TRANSFORMERS_AVAILABLE:
        if sentence_model is None:
            print("Loading sentence embedding model...")
            try:
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence embedding model loaded successfully!")
            except:
                print("⚠️ Failed to load sentence embedding model")
                
        if summarization_model is None:
            print("Loading text encoder for summarization...")
            try:
                summarization_model = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    device=0 if DEVICE == "cuda" else -1
                )
                print("✅ Summarization model loaded successfully!")
            except:
                print("⚠️ Failed to load summarization model, using fallback")
    
    return pipe

# Add legacy compatibility function
def load_model():
    """Legacy compatibility function."""
    return load_models()

class TextEncoder:
    """Advanced text encoder using transformer models for semantic understanding."""
    def __init__(self):
        # Don't capture sentence_model at init time - access it dynamically
        pass
        
    def encode_sentences(self, sentences):
        """Encode sentences into semantic vectors."""
        global sentence_model
        
        # Try to load sentence model if not already loaded
        if sentence_model is None and TRANSFORMERS_AVAILABLE:
            try:
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Sentence embedding model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Failed to load sentence embedding model: {e}")
        
        if sentence_model is not None:
            try:
                embeddings = sentence_model.encode(sentences)
                return embeddings
            except:
                pass
        
        # Fallback: TF-IDF based encoding
        print("⚠️ Using TF-IDF fallback for sentence encoding")
        if len(sentences) > 1:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            try:
                embeddings = vectorizer.fit_transform(sentences)
                return embeddings.toarray()
            except:
                pass
        
        # Basic fallback: word count vectors
        return self._basic_encode_sentences(sentences)
    
    def _basic_encode_sentences(self, sentences):
        """Basic encoding using word counts as fallback."""
        all_words = set()
        for sentence in sentences:
            words = sentence.lower().split()
            all_words.update(words)
        
        vocab = list(all_words)
        vectors = []
        
        for sentence in sentences:
            words = sentence.lower().split()
            vector = [words.count(word) for word in vocab]
            vectors.append(vector)
        
        return np.array(vectors)

class SemanticChunker:
    """Semantic-based text chunking using encoder-decoder architecture."""
    def __init__(self):
        self.encoder = TextEncoder()
        
    def extract_semantic_units(self, text):
        """Extract semantically meaningful units from text."""
        sentences = smart_sentence_split(text)
        if not sentences:
            return ["No content provided."]
        
        # Step 1: Analyze sentence types and importance
        analyzed_sentences = self._analyze_sentences(sentences)
        
        # Step 2: Group sentences by semantic similarity
        semantic_groups = self._group_by_semantics(analyzed_sentences)
        
        # Step 3: Create narrative chunks
        narrative_chunks = self._create_narrative_chunks(semantic_groups, text)
        
        return narrative_chunks
    
    def _analyze_sentences(self, sentences):
        """Analyze sentences for type, importance, and entities."""
        analyzed = []
        
        for sentence in sentences:
            analysis = {
                'text': sentence,
                'word_count': count_words(sentence),
                'entities': self._extract_entities(sentence),
                'sentence_type': self._classify_sentence_type(sentence),
                'importance_score': self._calculate_importance(sentence)
            }
            analyzed.append(analysis)
        
        return analyzed
    
    def _extract_entities(self, sentence):
        """Extract named entities from sentence."""
        entities = {'people': [], 'places': [], 'things': []}
        
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                tree = ne_chunk(pos_tags)
                
                for subtree in tree:
                    if isinstance(subtree, Tree):
                        entity_name = ' '.join([token for token, pos in subtree.leaves()])
                        if subtree.label() == 'PERSON':
                            entities['people'].append(entity_name)
                        elif subtree.label() in ['GPE', 'LOCATION']:
                            entities['places'].append(entity_name)
                        else:
                            entities['things'].append(entity_name)
                    else:
                        # Look for capitalized words that might be names
                        token, pos = subtree
                        if token[0].isupper() and pos in ['NN', 'NNP'] and len(token) > 2:
                            if token not in ['The', 'And', 'But', 'Or', 'So', 'Then']:
                                entities['things'].append(token)
            except:
                pass
        
        # Fallback: simple capitalized word detection
        words = sentence.split()
        for word in words:
            clean_word = word.strip('.,!?";:')
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                if clean_word.lower() not in ['the', 'and', 'but', 'or', 'so', 'then', 'he', 'she', 'it', 'they']:
                    entities['things'].append(clean_word)
        
        return entities
    
    def _classify_sentence_type(self, sentence):
        """Classify sentence type for better understanding."""
        sentence_lower = sentence.lower().strip()
        
        # Action sentences (high visual potential)
        action_keywords = ['ran', 'jumped', 'hit', 'threw', 'grabbed', 'pushed', 'pulled', 'opened', 'closed', 'walked', 'moved', 'attacked', 'defended']
        if any(keyword in sentence_lower for keyword in action_keywords):
            return 'action'
        
        # Dialogue
        if '"' in sentence or sentence.endswith('!') or sentence.endswith('?'):
            return 'dialogue'
        
        # Description/Setting
        description_keywords = ['was', 'were', 'had', 'stood', 'sat', 'looked', 'appeared', 'seemed']
        if any(keyword in sentence_lower for keyword in description_keywords):
            return 'description'
        
        # Transition
        transition_keywords = ['then', 'next', 'after', 'before', 'meanwhile', 'suddenly', 'finally']
        if any(sentence_lower.startswith(keyword) for keyword in transition_keywords):
            return 'transition'
        
        return 'narrative'
    
    def _calculate_importance(self, sentence):
        """Calculate importance score for sentence."""
        score = 0
        sentence_lower = sentence.lower()
        
        # Length factor (moderate length is better)
        word_count = count_words(sentence)
        if 5 <= word_count <= 15:
            score += 2
        elif 3 <= word_count <= 20:
            score += 1
        
        # Visual keywords
        visual_keywords = ['see', 'look', 'watch', 'appear', 'show', 'bright', 'dark', 'color', 'big', 'small', 'beautiful', 'ugly']
        score += sum(1 for keyword in visual_keywords if keyword in sentence_lower)
        
        # Action keywords
        action_keywords = ['fight', 'run', 'jump', 'fly', 'attack', 'escape', 'chase', 'catch']
        score += sum(2 for keyword in action_keywords if keyword in sentence_lower)
        
        # Emotion keywords
        emotion_keywords = ['happy', 'sad', 'angry', 'scared', 'excited', 'worried', 'surprised']
        score += sum(1 for keyword in emotion_keywords if keyword in sentence_lower)
        
        return score

    def _group_by_semantics(self, analyzed_sentences):
        """Group sentences by semantic similarity."""
        if len(analyzed_sentences) <= 1:
            return [analyzed_sentences]
        
        # Extract just the text for encoding
        sentences_text = [s['text'] for s in analyzed_sentences]
        
        try:
            # Get semantic embeddings
            embeddings = self.encoder.encode_sentences(sentences_text)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Group sentences based on similarity
            groups = []
            used_indices = set()
            
            for i, sentence in enumerate(analyzed_sentences):
                if i in used_indices:
                    continue
                
                # Start a new group
                group = [sentence]
                used_indices.add(i)
                
                # Find similar sentences
                for j in range(i + 1, len(analyzed_sentences)):
                    if j in used_indices:
                        continue
                    
                    # Check semantic similarity
                    if similarity_matrix[i][j] > 0.3:  # Threshold for similarity
                        group.append(analyzed_sentences[j])
                        used_indices.add(j)
                    
                    # Don't make groups too large
                    if len(group) >= 3:
                        break
                
                groups.append(group)
            
            return groups
            
        except Exception as e:
            print(f"⚠️ Semantic grouping failed: {e}, using fallback")
            # Fallback: group by sentence type and position
            return self._fallback_grouping(analyzed_sentences)
    
    def _fallback_grouping(self, analyzed_sentences):
        """Fallback grouping when semantic analysis fails."""
        groups = []
        current_group = []
        
        for sentence in analyzed_sentences:
            # Start new group if current is too large or type changes significantly
            if len(current_group) >= 2 or (
                current_group and 
                current_group[0]['sentence_type'] != sentence['sentence_type'] and
                sentence['sentence_type'] in ['action', 'dialogue']
            ):
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
            else:
                current_group.append(sentence)
        
        # Add remaining sentences
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _create_narrative_chunks(self, semantic_groups, original_text):
        """Create narrative chunks from semantic groups."""
        chunks = []
        
        for group in semantic_groups:
            # Combine sentences in the group
            combined_text = ' '.join([s['text'] for s in group])
            
            # Calculate chunk importance
            total_importance = sum(s['importance_score'] for s in group)
            avg_importance = total_importance / len(group)
            
            # Extract dominant entities
            all_entities = {'people': [], 'places': [], 'things': []}
            for sentence in group:
                for entity_type, entities in sentence['entities'].items():
                    all_entities[entity_type].extend(entities)
            
            # Remove duplicates and keep most common
            for entity_type in all_entities:
                all_entities[entity_type] = list(set(all_entities[entity_type]))
            
            # Create enhanced chunk
            chunk = {
                'text': combined_text,
                'importance': avg_importance,
                'entities': all_entities,
                'sentence_types': [s['sentence_type'] for s in group],
                'word_count': sum(s['word_count'] for s in group)
            }
            
            chunks.append(chunk)
        
        return chunks

def advanced_story_analysis(text):
    """Advanced story analysis using encoder-decoder architecture."""
    chunker = SemanticChunker()
    
    # Extract semantic chunks
    semantic_chunks = chunker.extract_semantic_units(text)
    
    # Convert chunks to panel text with optimization
    panels = []
    
    for chunk in semantic_chunks:
        # Determine if chunk should be split further
        if chunk['word_count'] > 25:  # Too long for single panel
            # Split by sentences within the chunk
            sentences = smart_sentence_split(chunk['text'])
            if len(sentences) > 1:
                # Split into smaller panels
                mid_point = len(sentences) // 2
                panel1 = ' '.join(sentences[:mid_point])
                panel2 = ' '.join(sentences[mid_point:])
                panels.append(panel1)
                panels.append(panel2)
            else:
                panels.append(chunk['text'])
        else:
            panels.append(chunk['text'])
    
    # Ensure minimum and maximum panel counts
    if len(panels) < MIN_PANELS:
        # Split longest panels
        while len(panels) < MIN_PANELS and any(count_words(p) > 10 for p in panels):
            longest_idx = max(range(len(panels)), key=lambda i: count_words(panels[i]))
            longest_panel = panels[longest_idx]
            sentences = smart_sentence_split(longest_panel)
            
            if len(sentences) > 1:
                mid_point = len(sentences) // 2
                panel1 = ' '.join(sentences[:mid_point])
                panel2 = ' '.join(sentences[mid_point:])
                panels[longest_idx] = panel1
                panels.insert(longest_idx + 1, panel2)
            else:
                break
    
    elif len(panels) > MAX_PANELS:
        # Combine similar panels
        while len(panels) > MAX_PANELS:
            # Find best panels to combine
            best_combo = None
            min_combined_length = float('inf')
            
            for i in range(len(panels) - 1):
                combined_length = count_words(panels[i]) + count_words(panels[i + 1])
                if combined_length < min_combined_length and combined_length <= 30:
                    min_combined_length = combined_length
                    best_combo = i
            
            if best_combo is not None:
                combined = panels[best_combo] + ' ' + panels[best_combo + 1]
                panels[best_combo] = combined
                panels.pop(best_combo + 1)
            else:
                break
    
    return panels

def smart_sentence_split(text):
    """Intelligently split text into sentences using NLTK or fallback method."""
    if NLTK_AVAILABLE:
        try:
            # Try NLTK first
            sentences = sent_tokenize(text)
        except:
            # Fallback to regex-based splitting
            sentences = re.split(r'[.!?]+', text)
    else:
        # Use regex-based splitting when NLTK is not available
        sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter sentences
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
    return sentences

def count_words(text):
    """Count words in text, with or without NLTK."""
    if NLTK_AVAILABLE:
        try:
            return len(word_tokenize(text))
        except:
            pass
    # Fallback word counting
    return len(text.split())

def analyze_story_structure(text):
    """Analyze the story and break it into meaningful panels."""
    sentences = smart_sentence_split(text)
    
    if not sentences:
        return ["No story provided."]
    
    total_words = sum(count_words(sentence) for sentence in sentences)
    
    # Calculate optimal number of panels based on content
    if total_words <= 20:
        target_panels = max(2, len(sentences))
    elif total_words <= 50:
        target_panels = max(3, min(4, len(sentences)))
    elif total_words <= 100:
        target_panels = max(4, min(6, math.ceil(total_words / OPTIMAL_WORDS_PER_PANEL)))
    else:
        target_panels = min(MAX_PANELS, math.ceil(total_words / OPTIMAL_WORDS_PER_PANEL))
    
    target_panels = max(MIN_PANELS, min(MAX_PANELS, target_panels))
    
    # Group sentences into panels
    if len(sentences) <= target_panels:
        # Each sentence gets its own panel
        panels = sentences
    else:
        # Group sentences to create meaningful panels
        panels = group_sentences_into_panels(sentences, target_panels)
    
    return panels

def group_sentences_into_panels(sentences, target_panels):
    """Group sentences into panels while maintaining narrative flow."""
    if len(sentences) <= target_panels:
        return sentences
    
    panels = []
    sentences_per_panel = len(sentences) / target_panels
    current_panel = []
    current_count = 0
    target_for_current = sentences_per_panel
    
    for i, sentence in enumerate(sentences):
        current_panel.append(sentence)
        current_count += 1
        
        # Decide when to close current panel
        should_close = (
            current_count >= target_for_current or  # Reached target count
            (len(panels) == target_panels - 1 and i == len(sentences) - 1) or  # Last panel
            (current_count >= 2 and len(panels) < target_panels - 1)  # Safe to close
        )
        
        if should_close:
            panels.append(' '.join(current_panel))
            current_panel = []
            current_count = 0
            # Adjust target for remaining panels
            remaining_sentences = len(sentences) - i - 1
            remaining_panels = target_panels - len(panels)
            if remaining_panels > 0:
                target_for_current = remaining_sentences / remaining_panels
    
    # Add any remaining sentences to the last panel
    if current_panel:
        if panels:
            panels[-1] += ' ' + ' '.join(current_panel)
        else:
            panels.append(' '.join(current_panel))
    
    return panels

def enhance_panel_description(panel_text):
    """Enhance panel text to be more visually descriptive for AI generation."""
    # Add visual action words - keep it concise
    enhanced_text = panel_text
    
    # Replace only the most important abstract concepts with visual ones
    replacements = {
        'was happy': 'smiled',
        'was sad': 'frowned',
        'was angry': 'scowled',
        'was scared': 'cowered',
        'discovered': 'found',
        'realized': 'understood'
    }
    
    for old, new in replacements.items():
        enhanced_text = enhanced_text.replace(old, new)
    
    return enhanced_text

def extract_characters_and_setting(story):
    """Extract main characters and setting from the story for consistency."""

    # Words that describe places/settings or adjectives used in place names —
    # these should NEVER be treated as a person's name even when capitalised.
    PLACE_WORDS = {
        # Geographical features
        'woods', 'wood', 'forest', 'forests', 'jungle', 'peak', 'peaks',
        'mountain', 'mountains', 'hill', 'hills', 'cliff', 'cliffs', 'valley',
        'beach', 'ocean', 'sea', 'lake', 'river', 'shore', 'bay', 'cape',
        'gulf', 'isle', 'island', 'islands', 'desert', 'plains', 'moor',
        'marsh', 'swamp', 'canyon', 'ridge', 'glacier', 'tundra',
        # Structures / places
        'castle', 'fortress', 'tower', 'palace', 'temple', 'shrine', 'ruins',
        'city', 'town', 'village', 'realm', 'kingdom', 'empire', 'land', 'world',
        'gate', 'road', 'path', 'bridge', 'wall', 'door', 'hall', 'manor',
        'harbour', 'harbor', 'port', 'mine', 'pit', 'garden', 'field', 'meadow',
        'cavern', 'cave', 'tunnel', 'underground', 'wasteland',
        # Common place-name adjectives (e.g. "Whispering Woods", "Obsidian Peaks")
        'whispering', 'obsidian', 'ancient', 'enchanted', 'haunted', 'sacred',
        'forbidden', 'northern', 'southern', 'eastern', 'western', 'great',
        'dark', 'golden', 'silver', 'crimson', 'black', 'white', 'deep', 'high',
        'old', 'new', 'lost', 'hidden', 'burning', 'frozen', 'eternal', 'shadowed',
        # Directional/generic adjectives that head place names
        'north', 'south', 'east', 'west', 'upper', 'lower', 'inner', 'outer',
    }

    # General stop-words that should never be treated as names
    STOP_WORDS = {
        'the', 'and', 'but', 'or', 'so', 'then', 'now', 'yet', 'still',
        'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their',
        'this', 'that', 'these', 'those', 'here', 'there',
        'for', 'with', 'from', 'into', 'onto', 'upon', 'over', 'under',
        'through', 'across', 'between', 'among', 'within', 'without',
        'just', 'only', 'even', 'also', 'too', 'very', 'quite', 'rather',
        'first', 'last', 'every', 'each', 'all', 'some', 'any', 'no', 'not',
        'more', 'most', 'both', 'few', 'many', 'much', 'such', 'same', 'other',
        'where', 'when', 'how', 'what', 'who', 'which',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    }

    name_counts = {}   # lowercase name → frequency

    sentences = smart_sentence_split(story)
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?";:—\u2014').lower()
            original   = word.strip('.,!?";:—\u2014')

            if not original or len(clean_word) <= 2:
                continue

            # --- Skip place/stop words immediately ---
            if clean_word in PLACE_WORDS or clean_word in STOP_WORDS:
                continue

            # --- Only consider capitalised words (proper nouns) ---
            if not original[0].isupper():
                continue

            # --- Skip words that follow an article → compound place names ---
            # e.g. "the Whispering Woods", "a Forbidden Forest"
            if i > 0:
                prev = words[i - 1].strip('.,!?";:—\u2014').lower()
                if prev in {'the', 'a', 'an'}:
                    continue

            # --- Skip if the *next* word is a known place word ---
            # e.g. "Whispering [Woods]", "Obsidian [Peaks]"
            if i < len(words) - 1:
                nxt = words[i + 1].strip('.,!?";:—\u2014').lower()
                if nxt in PLACE_WORDS:
                    continue

            # --- Skip sentence-starters that are just common words capitalised ---
            if i == 0 and clean_word in STOP_WORDS:
                continue

            # Looks like a genuine name → count it
            name_counts[clean_word] = name_counts.get(clean_word, 0) + 1

    # Sort by frequency: the most-mentioned name is the main character
    sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
    unique_characters = [name for name, _ in sorted_names[:3]]

    # --- Setting detection (unchanged logic, comprehensive list) ---
    setting_indicators = [
        'castle', 'palace', 'fortress', 'tower',
        'forest', 'woods', 'jungle', 'trees',
        'city', 'town', 'village', 'metropolis',
        'school', 'university', 'college', 'classroom',
        'house', 'home', 'cottage', 'mansion', 'apartment',
        'mountain', 'hill', 'cliff', 'valley',
        'beach', 'ocean', 'sea', 'lake', 'river',
        'space', 'spaceship', 'station', 'planet',
        'hospital', 'clinic', 'office', 'building',
        'laboratory', 'lab', 'workshop', 'factory',
        'garden', 'park', 'field', 'meadow',
        'cave', 'cavern', 'underground', 'tunnel',
        'island', 'desert', 'wasteland', 'ruins',
        'library', 'bookstore', 'shop', 'market',
        'tavern', 'inn', 'restaurant', 'cafe',
        'church', 'temple', 'cathedral', 'shrine',
    ]

    story_lower = story.lower()
    setting_counts = {}
    for indicator in setting_indicators:
        count = story_lower.count(indicator)
        if count > 0:
            setting_counts[indicator] = count

    # Sort by frequency so the most-prominent setting comes first
    sorted_settings = sorted(setting_counts.items(), key=lambda x: x[1], reverse=True)
    unique_settings = [s for s, _ in sorted_settings[:2]]

    return unique_characters, unique_settings

def create_enhanced_prompt_v2(panel_text, theme_style, characters, setting, panel_number, total_panels, entities=None):
    """Create an enhanced prompt using semantic understanding and entity information."""
    
    # Extract key visual elements from panel text
    visual_elements = extract_visual_elements(panel_text)
    
    # Get main character name if available
    main_char = characters[0] if characters else "character"
    
    # Get setting if available  
    location = setting[0] if setting else ""
    
    # Use entity information if provided
    if entities:
        # Prioritize entities from current panel
        panel_people = entities.get('people', [])
        panel_places = entities.get('places', [])
        panel_things = entities.get('things', [])
        
        if panel_people:
            main_char = panel_people[0]
        if panel_places and not location:
            location = panel_places[0]
    
    # Enhanced panel text with visual focus
    enhanced_text = enhance_panel_description_v2(panel_text, visual_elements)
    
    # Create style descriptor based on theme
    style_map = {
        'manga': "manga comic panel, anime style",
        'superhero': "superhero comic panel, dynamic pose", 
        'noir': "noir comic panel, dramatic shadows",
        'cartoon': "cartoon comic panel, animated style",
        'fantasy': "fantasy comic panel, magical atmosphere",
        'scifi': "sci-fi comic panel, futuristic setting",
        'webcomic': "modern comic panel, clean art"
    }
    
    style = next((v for k, v in style_map.items() if k in theme_style.lower()), "comic panel")
    
    # Build semantic-aware prompt
    prompt_parts = [
        enhanced_text[:40],  # Main action/scene
        style
    ]
    
    # Add character info
    if main_char and main_char != "character":
        prompt_parts.append(f"{main_char.lower()} character")
    
    # Add location info
    if location:
        prompt_parts.append(f"{location.lower()} background")
    
    # Add visual elements
    if visual_elements:
        prompt_parts.append(", ".join(visual_elements[:2]))
    
    prompt_parts.extend(["detailed illustration", "high quality"])
    
    full_prompt = ", ".join(prompt_parts)
    
    # Enhanced negative prompt
    negative_prompt = "blurry, low quality, text, watermark, signature, duplicate, copied artwork, bad anatomy"
    
    return full_prompt, negative_prompt

def extract_visual_elements(text):
    """Extract visual elements from text for better image generation."""
    visual_elements = []
    text_lower = text.lower()
    
    # Lighting and mood
    lighting_words = {
        'dark': 'dark atmosphere', 'bright': 'bright lighting', 
        'shadow': 'dramatic shadows', 'light': 'soft lighting',
        'sun': 'sunlight', 'moon': 'moonlight', 'fire': 'fire lighting'
    }
    
    for word, element in lighting_words.items():
        if word in text_lower:
            visual_elements.append(element)
            break
    
    # Colors
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gold', 'silver']
    for color in colors:
        if color in text_lower:
            visual_elements.append(f"{color} colors")
            break
    
    # Actions that translate well visually
    visual_actions = {
        'fight': 'action scene', 'run': 'running motion', 'jump': 'jumping pose',
        'fly': 'flying pose', 'fall': 'falling motion', 'climb': 'climbing pose',
        'hide': 'hiding pose', 'attack': 'attack pose', 'defend': 'defensive pose'
    }
    
    for action, visual in visual_actions.items():
        if action in text_lower:
            visual_elements.append(visual)
    
    # Emotions
    emotions = {
        'happy': 'happy expression', 'sad': 'sad expression', 
        'angry': 'angry expression', 'scared': 'fearful expression',
        'surprised': 'surprised expression', 'excited': 'excited expression'
    }
    
    for emotion, expression in emotions.items():
        if emotion in text_lower:
            visual_elements.append(expression)
    
    return visual_elements[:3]  # Limit to top 3 elements

def enhance_panel_description_v2(panel_text, visual_elements):
    """Enhanced panel description with semantic understanding."""
    enhanced_text = panel_text
    
    # More sophisticated replacements based on context
    replacements = {
        'was happy': 'smiled joyfully',
        'was sad': 'looked downcast', 
        'was angry': 'glared angrily',
        'was scared': 'trembled in fear',
        'was surprised': 'gasped in shock',
        'discovered': 'found and examined',
        'realized': 'suddenly understood',
        'went to': 'approached',
        'looked at': 'gazed upon',
        'thought about': 'pondered',
        'remembered': 'recalled vividly'
    }
    
    for old, new in replacements.items():
        enhanced_text = enhanced_text.replace(old, new)
    
    # Add action words where appropriate
    if any(word in enhanced_text.lower() for word in ['said', 'told', 'spoke']):
        if 'loudly' not in enhanced_text.lower():
            enhanced_text = enhanced_text.replace(' said', ' declared')
            enhanced_text = enhanced_text.replace(' told', ' announced')
    
    return enhanced_text

def calculate_optimal_layout(num_panels):
    """Calculate the best grid layout for the given number of panels."""
    if num_panels <= 2:
        return 2, 1  # 2 panels per row, 1 row
    elif num_panels <= 4:
        return 2, 2  # 2x2 grid
    elif num_panels <= 6:
        return 3, 2  # 3x2 grid
    elif num_panels <= 9:
        return 3, 3  # 3x3 grid
    else:
        return 4, math.ceil(num_panels / 4)  # 4 panels per row

def calculate_page_layout(total_panels):
    """Calculate how to distribute panels across multiple pages."""
    if total_panels <= MAX_PANELS_PER_PAGE:
        # Single page
        return [(total_panels, calculate_optimal_layout(total_panels))]
    
    # Multiple pages needed
    pages = []
    remaining_panels = total_panels
    
    while remaining_panels > 0:
        if remaining_panels <= MAX_PANELS_PER_PAGE:
            # Last page - use all remaining panels if >= MIN_PANELS_PER_PAGE
            if remaining_panels >= MIN_PANELS_PER_PAGE:
                panels_this_page = remaining_panels
            else:
                # If less than minimum, redistribute from previous page
                if pages:
                    # Take some panels from the last page
                    last_page_panels = pages[-1][0]
                    redistribute = min(2, last_page_panels - MIN_PANELS_PER_PAGE)
                    pages[-1] = (last_page_panels - redistribute, calculate_optimal_layout(last_page_panels - redistribute))
                    panels_this_page = remaining_panels + redistribute
                else:
                    panels_this_page = remaining_panels
        else:
            # Not the last page - use preferred number
            panels_this_page = PREFERRED_PANELS_PER_PAGE
        
        layout = calculate_optimal_layout(panels_this_page)
        pages.append((panels_this_page, layout))
        remaining_panels -= panels_this_page
    
    return pages

def generate_comic(story, theme='classic'):
    """Generate a comic strip using encoder-decoder architecture and advanced NLP."""
    
    # Check dependencies first
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("Cannot generate comic: Diffusers not available due to dependency conflicts. Please fix environment.")
    
    if not PIL_AVAILABLE:
        raise RuntimeError("Cannot generate comic: PIL not available.")
    
    load_models()  # Load all models including NLP models
    
    # Get theme style
    theme_info = COMIC_THEMES.get(theme, COMIC_THEMES['classic'])
    theme_style = theme_info['style']
    
    print(f"🤖 Starting advanced story analysis...")
    
    # Use advanced semantic analysis instead of basic analysis
    if TRANSFORMERS_AVAILABLE and sentence_model is not None:
        print("✅ Using encoder-decoder architecture for text analysis")
        panels = advanced_story_analysis(story)
    else:
        print("⚠️ Using fallback analysis (transformers not available)")
        panels = analyze_story_structure(story)  # Fallback to original method
    
    if not panels:
        raise ValueError("No valid content found in the story.")
    
    num_panels = len(panels)
    page_layouts = calculate_page_layout(num_panels)
    num_pages = len(page_layouts)
    
    # Extract characters and setting for consistency
    characters, setting = extract_characters_and_setting(story)
    
    # Enhanced semantic analysis of the story
    if TRANSFORMERS_AVAILABLE:
        story_context = analyze_story_context(story)
        print(f"📚 Story context: {story_context['theme']}, {story_context['mood']}")
    else:
        story_context = {'theme': 'general', 'mood': 'neutral'}
    
    print(f"📚 Semantic story analysis complete:")
    print(f"📊 Generated {num_panels} semantic panels across {num_pages} page(s)")
    print(f"🎨 Theme: {theme_info['name']}")
    if characters:
        print(f"👥 Main characters: {', '.join(characters)}")
    if setting:
        print(f"🏠 Setting: {', '.join(setting)}")
    
    generated_images = []

    # -------------------------
    # GENERATE PANELS WITH SEMANTIC UNDERSTANDING
    # -------------------------
    for i, panel_text in enumerate(panels):
        # Extract entities if we have semantic chunking
        panel_entities = None
        if TRANSFORMERS_AVAILABLE:
            chunker = SemanticChunker()
            panel_analysis = chunker._analyze_sentences([panel_text])
            if panel_analysis:
                panel_entities = panel_analysis[0]['entities']
        
        # Create enhanced prompt with semantic understanding
        enhanced_prompt, negative_prompt = create_enhanced_prompt_v2(
            panel_text, theme_style, characters, setting, 
            i+1, num_panels, panel_entities
        )

        print(f"🎬 Generating semantic panel {i+1}/{num_panels}: {panel_text[:50]}...")
        print(f"⚙️ Enhanced prompt: {enhanced_prompt[:80]}...")

        # Generate with enhanced parameters for semantic consistency
        import random
        
        # Determine device dynamically
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use story-aware seeding for consistency
        story_hash = hash(story[:100]) % 1000
        panel_seed = (story_hash + i * 137) % 100000  # More consistent seeding
        generator = torch.Generator(device=device).manual_seed(panel_seed)
        
        # Adjust steps based on device - fewer steps on GPU is still high quality
        num_steps = 25 if device == "cuda" else 30  # Reduced from 50 for speed
        
        # Enhanced generation parameters for better quality
        image = pipe(
            enhanced_prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,  # Optimized steps for quality vs speed
            guidance_scale=7.5,  # Standard guidance for good prompt adherence
            width=512,
            height=512,
            generator=generator,
            eta=0.0,  # More deterministic results
        ).images[0]

        draw = ImageDraw.Draw(image)

        # Dynamic text wrapping based on semantic content
        words_in_panel = count_words(panel_text)
        wrap_width = min(45, max(30, 50 - words_in_panel // 2))
        wrapped_text = textwrap.fill(panel_text, width=wrap_width)

        # Better text positioning
        bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (image.width - text_width) // 2
        y = image.height - text_height - MARGIN

        # Enhanced caption background
        padding = 8
        draw.rectangle(
            [(x - padding, y - padding), (x + text_width + padding, y + text_height + padding)],
            fill=(255, 255, 255, 240), outline="black", width=2
        )

        draw.text((x, y), wrapped_text, fill="black", font=font)

        image.save(f"{OUTPUT_DIR}/semantic_panel_{i+1}.png")
        generated_images.append(image)

    # -------------------------
    # COMBINE INTO MULTI-PAGE COMIC LAYOUT
    # -------------------------
    comic_paths = []
    imgs_base64 = []
    panel_idx = 0
    
    for page_num, (panels_on_page, (panels_per_row, num_rows)) in enumerate(page_layouts, 1):
        print(f"📄 Creating semantic page {page_num}/{num_pages} with {panels_on_page} panels ({panels_per_row}x{num_rows})")
        
        comic_width = panels_per_row * IMAGE_SIZE[0]
        comic_height = num_rows * IMAGE_SIZE[1]
        comic = Image.new("RGB", (comic_width, comic_height), "white")
        
        # Add panels for this page
        for local_idx in range(panels_on_page):
            if panel_idx < len(generated_images):
                img = generated_images[panel_idx]
                row = local_idx // panels_per_row
                col = local_idx % panels_per_row
                x_pos = col * IMAGE_SIZE[0]
                y_pos = row * IMAGE_SIZE[1]
                
                comic.paste(img, (x_pos, y_pos))
                
                # Add panel borders
                draw_comic = ImageDraw.Draw(comic)
                draw_comic.rectangle(
                    [x_pos, y_pos, x_pos + IMAGE_SIZE[0] - 1, y_pos + IMAGE_SIZE[1] - 1],
                    outline="black", width=3
                )
                panel_idx += 1
        
        # Save page
        comic_path = f"{OUTPUT_DIR}/semantic_comic_page_{page_num}.png"
        comic.save(comic_path)
        comic_paths.append(comic_path)
        
        # Convert to base64 for web display
        buffered = BytesIO()
        comic.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        imgs_base64.append(img_base64)

    layout_info = {
        'panels': num_panels,
        'pages': num_pages,
        'layout_per_page': [(panels, f"{layout[0]}x{layout[1]}") for panels, layout in page_layouts],
        'total_layout': f"{num_panels} semantic panels across {num_pages} page(s)",
        'analysis_method': 'Encoder-Decoder with Semantic Chunking' if TRANSFORMERS_AVAILABLE else 'Fallback Analysis'
    }

    print(f"\n✅ Multi-page semantic comic generated successfully!")
    print(f"📁 Saved {num_pages} page(s) at: {', '.join(comic_paths)}")
    print(f"📏 Layout: {layout_info['total_layout']}")
    print(f"🤖 Analysis method: {layout_info['analysis_method']}")
    
    return comic_paths, imgs_base64, num_panels, layout_info

def analyze_story_context(story):
    """Analyze overall story context for better generation."""
    context = {'theme': 'general', 'mood': 'neutral', 'genre': 'general'}
    
    story_lower = story.lower()
    
    # Detect theme
    if any(word in story_lower for word in ['magic', 'wizard', 'dragon', 'spell', 'enchanted']):
        context['theme'] = 'fantasy'
    elif any(word in story_lower for word in ['robot', 'space', 'alien', 'future', 'technology']):
        context['theme'] = 'sci-fi'
    elif any(word in story_lower for word in ['detective', 'crime', 'murder', 'mystery']):
        context['theme'] = 'mystery'
    elif any(word in story_lower for word in ['love', 'heart', 'romance', 'kiss']):
        context['theme'] = 'romance'
    elif any(word in story_lower for word in ['funny', 'laugh', 'joke', 'silly']):
        context['theme'] = 'comedy'
    
    # Detect mood
    if any(word in story_lower for word in ['dark', 'scary', 'fear', 'danger', 'death']):
        context['mood'] = 'dark'
    elif any(word in story_lower for word in ['bright', 'happy', 'joy', 'celebration', 'smile']):
        context['mood'] = 'cheerful'
    elif any(word in story_lower for word in ['sad', 'cry', 'loss', 'tragic']):
        context['mood'] = 'melancholic'
    elif any(word in story_lower for word in ['excited', 'adventure', 'thrilling', 'amazing']):
        context['mood'] = 'exciting'
    
    return context
    
# -------------------------
# ENHANCED TEXT PROCESSING DEMO (WORKS WITHOUT MODELS)
# -------------------------

def enhanced_text_analysis_demo(story):
    """Demonstrate enhanced text analysis capabilities."""
    print("🤖 Enhanced Text Analysis Demo:")
    print("=" * 50)
    
    # 1. Basic sentence analysis
    sentences = smart_sentence_split(story)
    print(f"📝 Sentence Splitting: {len(sentences)} sentences detected")
    
    # 2. Story context analysis  
    context = analyze_story_context(story)
    print(f"📚 Story Context: Theme='{context['theme']}', Mood='{context['mood']}'")
    
    # 3. Character and setting extraction
    characters, settings = extract_characters_and_setting(story)
    print(f"👥 Characters: {characters if characters else 'None detected'}")
    print(f"🏠 Settings: {settings if settings else 'None detected'}")
    
    # 4. Advanced story analysis
    try:
        panels = advanced_story_analysis(story)
        print(f"🔧 Semantic Panels: {len(panels)} meaningful chunks created")
        for i, panel in enumerate(panels[:3], 1):
            panel_text = panel['text'] if isinstance(panel, dict) else panel
            print(f"   Panel {i}: {panel_text[:60]}...")
    except Exception as e:
        print(f"⚠️ Advanced analysis error: {e}")
        
    return {
        'sentences': len(sentences),
        'context': context,
        'characters': characters,
        'settings': settings
    }

# -------------------------
# CLI MODE (optional)
# -------------------------
def run_cli():
    story = input("Enter your story :\n")
    comic_paths, _, num_panels, layout_info = generate_comic(story)
    print(f"\\nComic generated with {num_panels} panels across {len(comic_paths)} page(s):")
    for i, path in enumerate(comic_paths, 1):
        print(f"Page {i}: {path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Demo mode - show enhanced text processing
        test_story = """
        Once upon a time, in a magical forest, Luna discovered an ancient crystal.
        The crystal glowed with mysterious power and whispered ancient secrets.
        She faced a terrible dragon that had terrorized the kingdom for centuries.
        Using her wit and the crystal's magic, she befriended the lonely beast.
        Together they brought peace to the land and everyone celebrated.
        """
        
        result = enhanced_text_analysis_demo(test_story)
        print("\n✅ Enhanced text processing working perfectly!")
        print(f"📊 Analysis Results: {result}")
    elif len(sys.argv) > 1 and sys.argv[1] == 'cli':
        run_cli()
    else:
        print("🎨 Enhanced Comic Generator Module")
        print("📝 Run with 'python comic_generator.py demo' to see enhanced analysis")
        print("📱 Run 'python main.py' to start the web interface")
