import torch
import cv2
import numpy as np
import json
import os
import math
from datetime import datetime

# NLTK availability check
try:
    from nltk import word_tokenize, pos_tag, ne_chunk
    from nltk.tree import Tree
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available, using basic text processing")

# Handle diffusers import gracefully
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except Exception as e:
    DIFFUSERS_AVAILABLE = False
    print(f"⚠️ Diffusers import failed in video_generator: {e}")

# Optional video generation imports
try:
    from moviepy import ImageSequenceClip, CompositeVideoClip, VideoFileClip
    VIDEO_GENERATION_AVAILABLE = True
except ImportError:
    VIDEO_GENERATION_AVAILABLE = False
    print("⚠️ MoviePy not available, video generation disabled")

# Import enhanced functions from comic generator
from comic_generator import (
    smart_sentence_split, 
    load_models,
    extract_characters_and_setting,
    create_enhanced_prompt_v2,
    extract_visual_elements,
    enhance_panel_description_v2,
    SemanticChunker,
    TRANSFORMERS_AVAILABLE,
    analyze_story_context,
    DIFFUSERS_AVAILABLE  # Import this flag
)
# Note: pipe is imported dynamically after load_models() to get the actual loaded pipeline
import comic_generator

# -------------------------
# VIDEO CONFIG
# -------------------------
VIDEO_OUTPUT_DIR = "video_output"
FRAME_RATE = 25  # Increased for smoother video
IMAGE_DURATION = 2  # Reduced from 3 seconds
TRANSITION_DURATION = 0.5  # Faster transitions
VIDEO_RESOLUTION = (720, 512)  # Must be divisible by 8 for Stable Diffusion
# Comprehensive video style definitions
VIDEO_STYLES = {
    'cinematic': {
        'visual': 'cinematic film shot, professional movie lighting, film grain, realistic photography',
        'mood': 'dramatic cinematic atmosphere, film noir lighting',
        'camera': 'professional film camera work, shallow depth of field',
        'quality': 'hollywood movie quality, cinematic composition, photorealistic'
    },
    'animated': {
        'visual': 'high quality 3D animation, pixar movie style, studio ghibli quality, cinematic animation, detailed animated characters, professional animation studio',
        'mood': 'emotional animated atmosphere, cinematic lighting, rich colors',
        'camera': 'cinematic animated camera work, professional animation framing',
        'quality': 'pixar feature film quality, studio ghibli animation, high-end 3D animation, cinematic animated movie'
    },
    'realistic': {
        'visual': 'photorealistic, natural photography, realistic lighting, documentary style',
        'mood': 'natural realistic atmosphere, real world lighting',
        'camera': 'documentary style camera work, natural perspective',
        'quality': 'photographic realism, natural colors, real world'
    },
    'artistic': {
        'visual': 'artistic painting style, painterly, artistic illustration, fine art style',
        'mood': 'artistic creative atmosphere, painted lighting',
        'camera': 'artistic composition, painterly perspective',
        'quality': 'fine art painting, artistic masterpiece, illustration art'
    }
}

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# -------------------------
# VIDEO GENERATION FUNCTIONS
# -------------------------

def extract_story_timeline(story_text, style):
    """
    Extract cinematic scenes using Transformer/Encoder-Decoder architecture.
    Uses SemanticChunker, BART summarization, and NLP techniques.
    NO external API calls - fully local processing.
    """
    print("🧠 Using Transformer-based scene extraction (local NLP)...")
    
    # Determine scene count based on story length
    word_count = len(story_text.split())
    if word_count < 30:
        target_scenes = 3
    elif word_count < 80:
        target_scenes = 5
    elif word_count < 150:
        target_scenes = 7
    elif word_count < 300:
        target_scenes = 10
    else:
        target_scenes = min(15, math.ceil(word_count / 30))
    
    try:
        # Use transformer-based extraction
        return extract_scenes_with_transformers(story_text, target_scenes, style)
    except Exception as e:
        print(f"⚠️ Transformer extraction failed: {e}, using fallback")
        return create_timeline_fallback(story_text, target_scenes, style)


def extract_scenes_with_transformers(story_text, target_scenes, style):
    """
    Advanced scene extraction using encoder-decoder architecture:
    1. Semantic chunking with sentence embeddings
    2. BART/T5 summarization for scene descriptions
    3. NER for character/entity extraction
    4. TF-IDF for visual keyword extraction
    """
    from comic_generator import (
        SemanticChunker, 
        extract_characters_and_setting,
        sentence_model,
        summarization_model,
        NLTK_AVAILABLE
    )
    
    print(f"📊 Analyzing story structure for {target_scenes} scenes...")
    
    # Step 1: Extract global story context
    chars, settings = extract_characters_and_setting(story_text)
    main_char = chars[0] if chars else "protagonist"
    main_setting = settings[0] if settings else "scene"

    # Build a fixed character description reused in every scene prompt
    char_description = build_character_description(story_text, main_char)

    print(f"   👤 Main character: {main_char} ({char_description[:50]})")
    print(f"   🏠 Primary setting: {main_setting}")
    
    # Step 2: Semantic chunking using encoder
    chunker = SemanticChunker()
    semantic_chunks = chunker.extract_semantic_units(story_text)
    
    # Step 3: Distribute chunks into target scenes
    scenes = []
    chunks_per_scene = max(1, len(semantic_chunks) // target_scenes)
    
    for scene_idx in range(target_scenes):
        start_idx = scene_idx * chunks_per_scene
        end_idx = min(start_idx + chunks_per_scene, len(semantic_chunks))
        
        if start_idx >= len(semantic_chunks):
            break
            
        # Get chunks for this scene
        scene_chunks = semantic_chunks[start_idx:end_idx]
        scene_text = " ".join(scene_chunks) if isinstance(scene_chunks[0], str) else " ".join([c.get('text', str(c)) for c in scene_chunks])
        
        # Step 4: Generate scene description using summarization or extraction
        visual_desc = generate_scene_description(scene_text, main_char, main_setting, style)
        
        # Step 5: Extract mood from text
        mood = extract_mood(scene_text)
        
        # Step 6: Extract key visual elements
        key_elements = extract_key_elements(scene_text, main_char)
        
        # Step 7: Generate scene title from action/content
        scene_title = generate_scene_title(scene_text, scene_idx + 1, main_char)
        
        scenes.append({
            "scene_number": scene_idx + 1,
            "scene_title": scene_title,
            "visual_description": visual_desc,
            "mood": mood,
            "key_elements": key_elements
        })
    
    print(f"✅ Transformer extraction: {len(scenes)} cinematic scenes")
    for i, scene in enumerate(scenes[:3], 1):
        print(f"   Scene {i}: {scene['scene_title']} - {scene['visual_description'][:50]}...")
    
    return {
        "story_metadata": {
            "main_character": main_char,
            "char_description": char_description,
            "setting": main_setting,
            "visual_style": style
        },
        "scenes": scenes
    }


def generate_scene_description(scene_text, main_char, main_setting, style):
    """
    Generate visual scene description using NLP techniques.
    Uses summarization if available, otherwise extraction-based approach.
    """
    from comic_generator import summarization_model
    
    style_desc = VIDEO_STYLES.get(style, VIDEO_STYLES['cinematic'])
    
    # Try using BART/T5 summarization for scene description
    if summarization_model is not None:
        try:
            summary = summarization_model(
                scene_text,
                max_length=60,
                min_length=20,
                do_sample=False
            )[0]['summary_text']
            
            # Enhance with visual style
            visual_desc = f"{summary}, {style_desc['visual'][:30]}, featuring {main_char}"
            return visual_desc
        except:
            pass
    
    # Fallback: Extract key visual phrases from text
    return extract_visual_description(scene_text, main_char, main_setting, style_desc)


def extract_visual_description(scene_text, main_char, main_setting, style_desc):
    """Extract a *scene-specific* visual description using NLP + regex extraction.

    Captures compound phrases (brass key, salt spray, frozen waves) and
    scene-specific actions so every scene's prompt is meaningfully different.
    """
    import re as _re

    camera_angles = ["wide shot", "medium shot", "close-up shot", "establishing shot"]
    camera = camera_angles[abs(hash(scene_text[:30])) % len(camera_angles)]

    action_words     = []
    key_nouns        = []
    compound_phrases = []   # adjective+noun pairs e.g. "brass key", "salt spray"

    SKIP_VERBS = {
        'is', 'was', 'were', 'be', 'been', 'being',
        'have', 'had', 'has', 'do', 'did', 'does',
        'said', 'told', 'asked', 'replied', 'got', 'get', 'gets',
        'use', 'used', 'uses', 'let', 'lets', 'make', 'made',
        'come', 'came', 'goes', 'went', 'go',
    }
    SKIP_NOUNS = {
        'he', 'she', 'it', 'they', 'one', 'man', 'woman', 'person',
        'time', 'way', 'day', 'thing', 'part', 'place', 'year',
    }

    # --- NLTK-based extraction (preferred) ---
    sentences = smart_sentence_split(scene_text)
    if NLTK_AVAILABLE:
        try:
            from nltk import word_tokenize, pos_tag
            all_tokens = []
            for sent in sentences[:4]:
                tokens   = word_tokenize(sent)
                pos_tags = pos_tag(tokens)
                all_tokens.extend(pos_tags)

                for word, tag in pos_tags:
                    w = word.lower()
                    if tag.startswith('VB') and len(word) > 2 and w not in SKIP_VERBS:
                        action_words.append(w)
                    elif tag in ('NN', 'NNS', 'NNP', 'NNPS') and len(word) > 2:
                        if w not in SKIP_NOUNS:
                            key_nouns.append(w)

            # Extract compound adjective+noun phrases (e.g. "brass key", "salt spray")
            for j in range(len(all_tokens) - 1):
                w1, t1 = all_tokens[j]
                w2, t2 = all_tokens[j + 1]
                if (t1.startswith('JJ') and t2.startswith('NN')
                        and len(w1) > 2 and len(w2) > 2
                        and w2.lower() not in SKIP_NOUNS):
                    compound_phrases.append(f"{w1.lower()} {w2.lower()}")
        except Exception:
            pass

    # --- Priority: multi-word story-specific phrases via regex ---
    story_phrase_hits = _re.findall(
        r'\b(brass key|small key|old key|barnacle.encrusted rock|frozen waves|'
        r'salt spray|grandfather.s voice|faint glimmer|bruised plums|crashing waves|'
        r'freezing water|glass waves|obsidian peaks|whispering woods|brass legacy|'
        r'edge of the world|beneath the tide|carried on the wind)\b',
        scene_text.lower()
    )
    story_phrase_hits = list(dict.fromkeys(story_phrase_hits))

    # --- Regex fallback for bare verbs ---
    if not action_words:
        verb_hits = _re.findall(
            r'\b(stood|clutched|waded|turned|heard|noticed|searched|trekked|scaled|'
            r'froze|numbing|stinging|noticed|holding|reaching|inserted|discovered|'
            r'revealed|whispered|looked|gazed|walked|ran|fell|rose|opened|closed|'
            r'touched|grabbed|pulled|pushed|jumped|flew|stared|knelt|raised|lifted)\b',
            scene_text.lower()
        )
        action_words = list(dict.fromkeys(verb_hits))

    # --- Regex fallback for bare nouns ---
    if not key_nouns:
        noun_hits = _re.findall(
            r'\b(key|door|rock|tide|spray|keyhole|voice|wind|glimmer|sky|cliff|'
            r'light|cloud|mist|flame|fire|sword|map|letter|book|bridge|ship|'
            r'boat|sun|moon|star|stairs|window|mirror|shadow|chest|crown|ring|'
            r'stone|crystal|path|castle|tower|mountain|snow|rain|storm)\b',
            scene_text.lower()
        )
        key_nouns = list(dict.fromkeys(noun_hits))

    # --- Extract scene-local location phrase ---
    loc_matches = _re.findall(
        r'(?:at|in|on|beneath|under|through|across|into|upon|over|near|along)\s+'
        r'(?:the\s+)?([\w]+(?:\s+[\w]+){0,3})',
        scene_text.lower()
    )
    LOCATION_STOP = {'he', 'she', 'it', 'his', 'her', 'the', 'a', 'an',
                     main_char.lower(), main_setting.lower()}
    scene_location = main_setting
    for loc in loc_matches:
        loc = loc.strip()
        if loc and loc.split()[0] not in LOCATION_STOP and len(loc) > 3:
            scene_location = loc
            break

    # --- Build the final description ---
    action_phrase = action_words[0] if action_words else "observing"
    desc_parts = [f"{camera} of {main_char} {action_phrase}"]

    if scene_location:
        desc_parts.append(scene_location)

    # Compound phrases take priority (most descriptive)
    detail_items = []
    for p in (story_phrase_hits + compound_phrases)[:4]:
        if p not in detail_items:
            detail_items.append(p)
    # Fill remaining slots with bare nouns
    for n in key_nouns:
        if n.lower() not in {main_char.lower(), main_setting.lower(),
                             'the', 'a', 'an', 'and', 'or'} and n not in detail_items:
            detail_items.append(n)
        if len(detail_items) >= 4:
            break
    if detail_items:
        desc_parts.append(', '.join(detail_items[:4]))

    desc_parts.append(style_desc['visual'][:45])

    return ', '.join(desc_parts)


def build_character_description(story_text, char_name):
    """Build a fixed, consistent visual description of the main character.
    
    The SAME string is injected into every scene prompt so Stable Diffusion
    generates a visually consistent person across all frames.
    """
    import re as _re
    text_lower = story_text.lower()

    # --- Gender ---
    if sum(text_lower.count(w) for w in ['he ', 'his ', 'him ']) >= \
       sum(text_lower.count(w) for w in ['she ', 'her ', 'hers ']):
        gender = 'man'
    else:
        gender = 'woman'

    # --- Age ---
    if any(w in text_lower for w in ['boy ', 'girl ', 'child ', 'teen', 'young']):
        age = 'young'
    elif any(w in text_lower for w in ['old ', 'elderly', 'aged ', 'ancient']):
        age = 'elderly'
    else:
        age = 'adult'

    # --- Hair ---
    hair_map = [
        ('dark hair', ['dark hair', 'black hair', 'dark-haired']),
        ('blond hair', ['blond', 'blonde', 'golden hair']),
        ('red hair',   ['red hair', 'auburn', 'ginger']),
        ('grey hair',  ['grey hair', 'gray hair', 'silver hair', 'white hair']),
        ('brown hair', ['brown hair', 'chestnut']),
    ]
    hair = 'dark brown hair'  # default
    for desc, keywords in hair_map:
        if any(k in text_lower for k in keywords):
            hair = desc
            break

    # --- Clothing (scan story for hints) ---
    clothing_map = [
        ('heavy coat',   ['coat', 'overcoat', 'trench coat']),
        ('dark cloak',   ['cloak', 'cape']),
        ('jacket',       ['jacket']),
        ('robe',         ['robe']),
        ('armor',        ['armor', 'armour']),
        ('simple shirt and trousers', []),   # safe default
    ]
    clothing = 'simple shirt and trousers'  # default
    for desc, keywords in clothing_map:
        if keywords and any(k in text_lower for k in keywords):
            clothing = desc
            break

    # --- Story-specific props visible on the character ---
    props = []
    if _re.search(r'\b(key|brass key|small key)\b', text_lower):
        props.append('holding a small brass key')

    # --- Assemble ---
    parts = [f'{age} {gender}', hair, clothing]
    if props:
        parts.append(props[0])   # only first prop to keep prompt tight

    return ', '.join(parts)


def extract_mood(scene_text):
    """Extract emotional mood from scene text using sentiment analysis."""
    scene_lower = scene_text.lower()
    
    # Mood keywords mapping
    mood_mappings = {
        'joyful': ['happy', 'joy', 'laugh', 'smile', 'bright', 'cheerful', 'delight'],
        'melancholic': ['sad', 'sorrow', 'tear', 'cry', 'lonely', 'grief', 'mourn'],
        'tense': ['danger', 'threat', 'fear', 'afraid', 'nervous', 'worried', 'anxious'],
        'mysterious': ['strange', 'odd', 'mysterious', 'curious', 'secret', 'hidden', 'unknown'],
        'peaceful': ['calm', 'quiet', 'serene', 'peaceful', 'gentle', 'soft', 'tranquil'],
        'dramatic': ['sudden', 'shock', 'gasp', 'realize', 'discover', 'reveal'],
        'romantic': ['love', 'heart', 'tender', 'embrace', 'kiss', 'passion'],
        'adventurous': ['journey', 'quest', 'explore', 'discover', 'venture', 'brave']
    }
    
    # Score each mood
    mood_scores = {}
    for mood, keywords in mood_mappings.items():
        score = sum(1 for kw in keywords if kw in scene_lower)
        if score > 0:
            mood_scores[mood] = score
    
    # Return highest scoring mood, or default
    if mood_scores:
        return max(mood_scores, key=mood_scores.get)
    
    return "dramatic"


def extract_key_elements(scene_text, main_char):
    """Extract key visual elements using NER and noun extraction."""
    elements = [main_char]
    
    if NLTK_AVAILABLE:
        try:
            from nltk import word_tokenize, pos_tag, ne_chunk
            from nltk.tree import Tree
            
            tokens = word_tokenize(scene_text)
            pos_tags = pos_tag(tokens)
            tree = ne_chunk(pos_tags)
            
            for subtree in tree:
                if isinstance(subtree, Tree):
                    entity = ' '.join([token for token, _ in subtree.leaves()])
                    if entity not in elements and len(entity) > 2:
                        elements.append(entity)
                elif subtree[1] in ['NN', 'NNS', 'NNP']:
                    word = subtree[0]
                    if word[0].isupper() and word not in elements and len(word) > 2:
                        elements.append(word)
        except:
            pass
    
    # Fallback: extract capitalized words
    if len(elements) < 3:
        words = scene_text.split()
        for word in words:
            clean = word.strip('.,!?"\'')
            if clean and clean[0].isupper() and clean not in elements and len(clean) > 2:
                if clean.lower() not in ['the', 'and', 'but', 'then', 'when', 'where']:
                    elements.append(clean)
                    if len(elements) >= 5:
                        break
    
    return elements[:5]


def generate_scene_title(scene_text, scene_num, main_char):
    """Generate descriptive scene title from text content."""
    
    # Try to extract the main action or event
    sentences = smart_sentence_split(scene_text)
    if not sentences:
        return f"Scene {scene_num}"
    
    first_sentence = sentences[0]
    
    # Extract key action verb and object
    if NLTK_AVAILABLE:
        try:
            from nltk import word_tokenize, pos_tag
            
            tokens = word_tokenize(first_sentence)
            pos_tags = pos_tag(tokens)
            
            verb = None
            obj = None
            
            for word, tag in pos_tags:
                if tag.startswith('VB') and not verb and len(word) > 2:
                    verb = word
                elif tag in ['NN', 'NNS', 'NNP'] and not obj and len(word) > 2:
                    obj = word
            
            if verb and obj:
                # Convert verb to gerund form for title
                if verb.endswith('e'):
                    verb_ing = verb[:-1] + 'ing'
                elif verb.endswith('ing'):
                    verb_ing = verb
                else:
                    verb_ing = verb + 'ing'
                
                return f"{main_char} {verb_ing.title()} the {obj.title()}"
            elif verb:
                return f"{main_char}'s {verb.title()}"
        except:
            pass
    
    # Fallback: use first few words
    words = first_sentence.split()[:5]
    title = ' '.join(words)
    if len(title) > 30:
        title = title[:30] + "..."
    
    return title if title else f"Scene {scene_num}"


# Import NLTK_AVAILABLE check
try:
    from comic_generator import NLTK_AVAILABLE
except:
    NLTK_AVAILABLE = False

def create_timeline_fallback(story_text, target_scenes, style):
    """
    FALLBACK: Splits story text into segments and attempts basic visual inference.
    Outputs the same structure as Director's Cut extraction.
    """
    print(f"🔧 Using fallback with basic visual inference for {target_scenes} scenes")
    
    # Simple consistent split
    sentences = smart_sentence_split(story_text)
    
    total_sentences = len(sentences)
    if total_sentences == 0:
        sentences = [story_text]
        total_sentences = 1
        
    # Calculate chunk size
    chunk_size = max(1, math.ceil(total_sentences / target_scenes))
    
    scenes = []
    
    # Extract global entities for context
    chars, settings = extract_characters_and_setting(story_text)
    main_char = chars[0] if chars else "character"
    main_setting = settings[0] if settings else "setting"
    char_description = build_character_description(story_text, main_char)
    
    for i in range(target_scenes):
        start = i * chunk_size
        if start >= total_sentences:
            break
            
        end = min(start + chunk_size, total_sentences)
        chunk = sentences[start:end]
        segment_text = " ".join(chunk)
        
        # Local entity extraction (basic)
        local_chars, local_settings = extract_characters_and_setting(segment_text)
        current_location = local_settings[0] if local_settings else main_setting
        
        # Basic visual inference from text
        # Look for action words, descriptive phrases
        visual_keywords = []
        if any(word in segment_text.lower() for word in ['stopped', 'froze', 'paused']):
            visual_keywords.append("frozen moment")
        if any(word in segment_text.lower() for word in ['light', 'sun', 'glow', 'shine']):
            visual_keywords.append("lighting effects")
        if any(word in segment_text.lower() for word in ['dark', 'shadow', 'night']):
            visual_keywords.append("dark atmosphere")
            
        # Construct a basic visual description
        visual_desc = f"{segment_text[:80]}"  # Use the actual text as base
        if visual_keywords:
            visual_desc += f", with {', '.join(visual_keywords)}"
        
        # Infer mood from text sentiment
        mood = "dramatic"
        if any(word in segment_text.lower() for word in ['smiled', 'laughed', 'happy', 'joy']):
            mood = "joyful"
        elif any(word in segment_text.lower() for word in ['strange', 'odd', 'mysterious', 'curious']):
            mood = "mysterious"
        elif any(word in segment_text.lower() for word in ['sad', 'cry', 'tear', 'sorrow']):
            mood = "melancholic"
        
        scenes.append({
            "scene_number": i + 1,
            "scene_title": f"Scene {i + 1}",
            "visual_description": visual_desc,
            "mood": mood,
            "key_elements": [main_char, current_location]
        })
    
    return {
        "story_metadata": {
            "main_character": main_char,
            "char_description": char_description,
            "setting": main_setting,
            "visual_style": style
        },
        "scenes": scenes
    }


def create_enhanced_video_prompts(current_scene, previous_scene, story_metadata, scene_index, total_scenes, style):
    """Generate enhanced image prompts using encoder-decoder architecture and semantic understanding."""
    import re as _re

    # Get style definitions
    style_info = VIDEO_STYLES.get(style, VIDEO_STYLES['cinematic'])

    # Extract story metadata (fallback values)
    main_character = story_metadata.get('main_character', 'character')
    setting        = story_metadata.get('setting', 'environment')

    # Extract current scene details with type safety
    if not isinstance(current_scene, dict):
        current_scene = {'visual_description': 'scene action', 'mood': 'dramatic', 'key_elements': []}

    visual_description = current_scene.get('visual_description', 'scene in progress')
    scene_title        = current_scene.get('scene_title', f'Scene {scene_index + 1}')
    mood               = current_scene.get('mood', 'dramatic')

    # ---------------------------------------------------------------
    # 1. HUMAN CHARACTER ANCHOR  (consistent across EVERY scene)
    #    Use the fixed char_description stored in story_metadata so SD
    #    generates the same visual appearance in every frame.
    # ---------------------------------------------------------------
    char_description = story_metadata.get('char_description', '')
    if char_description:
        char_anchor = char_description          # e.g. "adult man, dark brown hair, heavy coat, holding a brass key"
    elif main_character and main_character not in ('character', 'protagonist'):
        char_anchor = f"adult man named {main_character}"
    else:
        char_anchor = "adult man"

    # ---------------------------------------------------------------
    # 2. EXTRACT SCENE-SPECIFIC ACTIONS + OBJECTS
    #    Pull the meaningful words out of the visual_description that
    #    was already built by extract_visual_description().
    # ---------------------------------------------------------------
    # Strip camera prefix (e.g. "close-up shot of young adult man stood, ")
    desc_body = _re.sub(
        r'^(wide shot|medium shot|close-up shot|establishing shot)\s+of\s+[^,]+,\s*',
        '', visual_description, flags=_re.IGNORECASE
    ).strip(', ')

    # Split off the style suffix (style info starts at "high quality" / "pixar")
    style_cutoff = _re.search(
        r',\s*(high quality|pixar|cinematic animation|3d animation|photorealistic|'
        r'artistic painting|film grain|studio ghibli)',
        desc_body, flags=_re.IGNORECASE
    )
    if style_cutoff:
        scene_specific = desc_body[:style_cutoff.start()].strip(', ')
    else:
        scene_specific = desc_body[:80].strip(', ')

    # ---------------------------------------------------------------
    # 3. SAFE SETTING DESCRIPTION
    #    Never use raw words like "ocean" or "sea" as the *subject* —
    #    anchor them as a background / environment phrase instead.
    # ---------------------------------------------------------------
    WATER_WORDS = {'ocean', 'sea', 'lake', 'river', 'water', 'tide', 'shore', 'coast',
                   'beach', 'waves', 'surf', 'bay', 'harbour', 'harbor'}

    # Also scrub water/creature trigger words from scene_specific so they
    # don't appear inline and override the human anchor.
    CREATURE_TRIGGER_WORDS = WATER_WORDS | {
        'barnacle', 'barnacles', 'seaweed', 'coral', 'abyss', 'depths',
        'deep', 'underwater', 'submerged', 'sunken', 'aquatic',
    }
    scene_specific_parts = [
        token for token in scene_specific.split(', ')
        if token.strip().lower() not in CREATURE_TRIGGER_WORDS
    ]
    scene_specific = ', '.join(scene_specific_parts)

    if setting and setting.lower() in WATER_WORDS:
        setting_phrase = f"coastal shoreline background"
    elif setting and setting not in ('environment', 'scene'):
        setting_phrase = f"{setting} environment background"
    else:
        setting_phrase = ""

    # ---------------------------------------------------------------
    # 4. STYLE KEYWORDS (short excerpt)
    # ---------------------------------------------------------------
    style_keywords = style_info['visual'][:50]
    quality        = style_info['quality'][:40]

    # ---------------------------------------------------------------
    # 5. PROGRESSIVE CAMERA
    # ---------------------------------------------------------------
    scene_progress = scene_index / max(1, total_scenes - 1)
    if scene_progress < 0.15:
        camera_note = "establishing wide shot"
    elif scene_progress > 0.85:
        camera_note = "dramatic close-up"
    else:
        camera_note = "medium narrative shot"

    # ---------------------------------------------------------------
    # 6. ASSEMBLE FINAL PROMPT
    #    Order matters for SD: subject first, then action, then setting.
    # ---------------------------------------------------------------
    parts = [
        "masterpiece, best quality, ultra detailed, sharp focus",  # quality boosters FIRST
        char_anchor,          # who — fixed appearance anchor
        scene_specific,       # what they are doing + objects
        setting_phrase,       # where (safe phrasing)
        f"{mood} atmosphere",
        style_keywords,
        quality,
        camera_note,
        "cinematic framing",
    ]
    video_enhanced_prompt = ", ".join(p for p in parts if p)

    # ---------------------------------------------------------------
    # 7. COMPREHENSIVE NEGATIVE PROMPT
    #    Explicitly ban creatures, sea life, monsters, animals.
    # ---------------------------------------------------------------
    creature_negative = (
        "sea monster, sea creature, ocean monster, water monster, kraken, leviathan, "
        "sea serpent, mermaid, merman, fish, shark, whale, dolphin, octopus, squid, "
        "crab, jellyfish, underwater creature, aquatic creature, aquatic animal, "
        "fantasy creature, mythical creature, monster, beast, animal, wildlife, "
        "alien creature, dragon, demon, horror creature, tentacles, scales, fins, gills"
    )
    base_negative = (
        "low quality, blurry, distorted, deformed, disfigured, ugly, bad anatomy, "
        "wrong anatomy, extra limbs, missing limbs, floating limbs, mutation, mutated, "
        "bad proportions, duplicate, cropped, worst quality, jpeg artifacts, "
        "signature, watermark, username, text, error, malformed"
    )
    # Style-specific negatives (prevent SD from mixing styles)
    style_negatives_map = {
        'cinematic':  "cartoon, animated, anime, 3D render, cel-shaded, disney, pixar, illustration, drawing, sketch, toon, comic",
        'animated':   "photorealistic, live action, realistic photography, film grain, real person, natural lighting, photo, photograph, real life, childish cartoon, simple cartoon, stick figure, flat colors",
        'realistic':  "cartoon, animated, anime, 3D render, cel-shaded, disney, pixar, illustration, drawing, sketch, artistic, painted, toon, comic",
        'artistic':   "photorealistic, live action, realistic photography, film grain, real person, natural lighting, photo, photograph",
    }
    style_negative = style_negatives_map.get(style, style_negatives_map['cinematic'])

    video_negative_prompt = f"{creature_negative}, {style_negative}, {base_negative}, static image, still photo, amateur quality"

    return video_enhanced_prompt, video_negative_prompt

def create_fallback_scenes(story_text):
    """Fallback scene creation if Gemini API fails."""
    sentences = smart_sentence_split(story_text)
    
    scenes = []
    for i, sentence in enumerate(sentences[:6]):  # Max 6 scenes
        scenes.append({
            "description": sentence[:50],  # Limit description length
            "environment": "general setting",
            "action": "scene unfolds",
            "mood": "neutral",
            "objects": ["character", "environment"],
            "focus": "main action"
        })
    
    return {
        "summary": {
            "main_characters": ["character"],
            "setting": "story setting", 
            "theme": "narrative",
            "mood": "storytelling"
        },
        "scenes": scenes
    }

def normalize_image_prompt(scene_data, story_summary, style="cinematic", max_length=75):
    """Create clean, controlled prompts with strict constraints."""
    
    # Extract core elements with length limits
    action = scene_data.get('action', 'scene')[:30]  # Max 30 chars
    environment = scene_data.get('environment', 'setting')[:25]  # Max 25 chars
    mood = scene_data.get('mood', 'neutral')[:15]  # Max 15 chars
    focus = scene_data.get('focus', 'center')[:20]  # Max 20 chars
    
    # Global style consistency
    style_templates = {
        'cinematic': 'cinematic film shot',
        'animated': 'animated cartoon style', 
        'realistic': 'photorealistic scene',
        'artistic': 'artistic illustration'
    }
    
    base_style = style_templates.get(style, 'cinematic film shot')
    
    # Character consistency - use same character descriptor throughout
    main_char = story_summary.get('main_characters', ['person'])[0]
    char_descriptor = f"consistent {main_char}"
    
    # Build core prompt with strict word limits
    core_elements = [
        base_style,
        action,
        environment, 
        char_descriptor,
        f"{mood} mood",
        focus,
        "high quality"
    ]
    
    # Join and ensure under token limit
    prompt = ", ".join(core_elements)
    
    # Truncate if too long (CLIP has 77 token limit)
    words = prompt.split()
    if len(words) > 15:  # Conservative limit to avoid token overflow
        prompt = " ".join(words[:15])
    
    return prompt

def create_negative_prompt():
    """Standard negative prompt to avoid clutter and ensure quality."""
    return "blurry, low quality, distorted, multiple scenes, cluttered, text, watermark, duplicated objects, confusing composition, poor lighting, artifacts"

def generate_scene_variations_improved(scene_data, story_summary, style_theme="cinematic"):
    """Generate clean, consistent visual variations for each scene."""
    base_prompt = normalize_image_prompt(scene_data, story_summary, style_theme)
    
    # Create subtle variations while maintaining consistency
    variations = [
        {
            "prompt": f"wide shot, {base_prompt}, establishing view",
            "camera": "wide establishing",
            "motion": "static"
        },
        {
            "prompt": f"medium shot, {base_prompt}, character focus",
            "camera": "medium focus", 
            "motion": "slight zoom"
        },
        {
            "prompt": f"close detail, {base_prompt}, emotional moment",
            "camera": "close detail",
            "motion": "slow zoom"
        }
    ]
    
    return variations

def fallback_scene_breakdown(text, target_scenes=5):
    """Fallback function when Gemini API is not available."""
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    scenes = []
    sentences_per_scene = max(1, len(sentences) // target_scenes)
    
    for i in range(0, len(sentences), sentences_per_scene):
        scene_sentences = sentences[i:i+sentences_per_scene]
        scenes.append({
            "scene_number": len(scenes) + 1,
            "description": " ".join(scene_sentences),
            "visual_elements": "characters, setting, action",
            "action": "story progression",
            "environment": "story setting"
        })
    
    return {
        "summary": {
            "main_characters": ["main character"],
            "setting": "story setting", 
            "theme": "narrative"
        },
        "scenes": scenes
    }

def normalize_image_prompt(scene_data, story_summary, style="cinematic", max_length=75):
    """Create clean, controlled prompts with strict constraints."""
    
    # Extract core elements with length limits
    action = scene_data.get('action', 'scene')[:30]  # Max 30 chars
    environment = scene_data.get('environment', 'setting')[:25]  # Max 25 chars
    mood = scene_data.get('mood', 'neutral')[:15]  # Max 15 chars
    focus = scene_data.get('focus', 'center')[:20]  # Max 20 chars
    
    # Global style consistency
    style_templates = {
        'cinematic': 'cinematic film shot',
        'animated': 'animated cartoon style', 
        'realistic': 'photorealistic scene',
        'artistic': 'artistic illustration'
    }
    
    base_style = style_templates.get(style, 'cinematic film shot')
    
    # Character consistency - use same character descriptor throughout
    main_char = story_summary.get('main_characters', ['person'])[0]
    char_descriptor = f"consistent {main_char}"
    
    # Build core prompt with strict word limits
    core_elements = [
        base_style,
        action,
        environment, 
        char_descriptor,
        f"{mood} mood",
        focus,
        "high quality"
    ]
    
    # Join and ensure under token limit
    prompt = ", ".join(core_elements)
    
    # Truncate if too long (CLIP has 77 token limit)
    words = prompt.split()
    if len(words) > 15:  # Conservative limit to avoid token overflow
        prompt = " ".join(words[:15])
    
    return prompt

def create_negative_prompt():
    """Standard negative prompt to avoid clutter and ensure quality."""
    return "blurry, low quality, distorted, multiple scenes, cluttered, text, watermark, duplicated objects, confusing composition, poor lighting, artifacts"

def generate_scene_variations_improved(scene_data, story_summary, style_theme="cinematic"):
    """Generate clean, consistent visual variations for each scene."""
    base_prompt = normalize_image_prompt(scene_data, story_summary, style_theme)
    
    # Create subtle variations while maintaining consistency
    variations = [
        {
            "prompt": f"wide shot, {base_prompt}, establishing view",
            "camera": "wide establishing",
            "motion": "static"
        },
        {
            "prompt": f"medium shot, {base_prompt}, character focus",
            "camera": "medium focus", 
            "motion": "slight zoom"
        },
        {
            "prompt": f"close detail, {base_prompt}, emotional moment",
            "camera": "close detail",
            "motion": "slow zoom"
        }
    ]
    
    return variations



def apply_animation_effects(image_path, effect_type, duration=3):
    """Apply Ken Burns and other animation effects to images."""
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    h, w = img.shape[:2]
    target_w, target_h = VIDEO_RESOLUTION
    
    frames = []
    fps = FRAME_RATE
    total_frames = int(duration * fps)
    
    # Resize image to be larger than target for animation effects
    scale_factor = 1.3
    scaled_w = int(target_w * scale_factor)
    scaled_h = int(target_h * scale_factor)
    img_scaled = cv2.resize(img, (scaled_w, scaled_h))
    
    for frame_num in range(total_frames):
        progress = frame_num / (total_frames - 1) if total_frames > 1 else 0
        
        if effect_type == "zoom_in":
            # Zoom in effect
            start_scale = 1.0
            end_scale = 1.2
            current_scale = start_scale + (end_scale - start_scale) * progress
            
            crop_w = int(target_w / current_scale)
            crop_h = int(target_h / current_scale)
            
            start_x = (scaled_w - crop_w) // 2
            start_y = (scaled_h - crop_h) // 2
            
            cropped = img_scaled[start_y:start_y+crop_h, start_x:start_x+crop_w]
            frame = cv2.resize(cropped, (target_w, target_h))
            
        elif effect_type == "zoom_out":
            # Zoom out effect
            start_scale = 1.2
            end_scale = 1.0
            current_scale = start_scale + (end_scale - start_scale) * progress
            
            crop_w = int(target_w / current_scale)
            crop_h = int(target_h / current_scale)
            
            start_x = (scaled_w - crop_w) // 2
            start_y = (scaled_h - crop_h) // 2
            
            cropped = img_scaled[start_y:start_y+crop_h, start_x:start_x+crop_w]
            frame = cv2.resize(cropped, (target_w, target_h))
            
        elif effect_type == "pan_right":
            # Pan right effect
            max_offset = scaled_w - target_w
            offset_x = int(max_offset * progress)
            
            cropped = img_scaled[0:target_h, offset_x:offset_x+target_w]
            frame = cropped
            
        elif effect_type == "pan_left":
            # Pan left effect  
            max_offset = scaled_w - target_w
            offset_x = max_offset - int(max_offset * progress)
            
            cropped = img_scaled[0:target_h, offset_x:offset_x+target_w]
            frame = cropped
            
        else:  # static
            # Static with slight movement
            offset_x = int((scaled_w - target_w) // 2)
            offset_y = int((scaled_h - target_h) // 2)
            frame = img_scaled[offset_y:offset_y+target_h, offset_x:offset_x+target_w]
            
        frames.append(frame)
    
    return frames

def create_video_with_transitions(frames_list, num_scenes):
    """Create final video with smooth transitions."""
    if not frames_list:
        raise ValueError("No frames to create video")
    
    # Convert frames to temporary video files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(VIDEO_OUTPUT_DIR, f"story_video_{timestamp}.mp4")
    
    print(f"🎞️ Creating video with {len(frames_list)} frames at {video_path}")
    
    # Try different codecs for better web compatibility - avoid OpenH264
    codecs_to_try = [
        cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4 Part 2 - most compatible
        cv2.VideoWriter_fourcc(*'XVID'),  # Xvid - good compatibility 
        cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG - always works
        cv2.VideoWriter_fourcc('M','J','P','G'),  # Alternative MJPG
    ]
    
    out = None
    for i, fourcc in enumerate(codecs_to_try):
        try:
            print(f"📝 Trying codec {i+1}/{len(codecs_to_try)}...")
            out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, VIDEO_RESOLUTION)
            if out.isOpened():
                print(f"✅ Codec {i+1} successful")
                break
            else:
                out.release()
                out = None
        except Exception as e:
            print(f"❌ Codec {i+1} failed: {e}")
            if out:
                out.release()
            out = None
    
    if not out or not out.isOpened():
        raise RuntimeError("Failed to create video writer with any codec")
    
    # Add crossfade transitions between scenes
    frames_per_scene = len(frames_list) // num_scenes if num_scenes > 0 else len(frames_list)
    transition_frames = int(TRANSITION_DURATION * FRAME_RATE)
    
    frames_written = 0
    for i, frame in enumerate(frames_list):
        # Validate frame
        if frame is None:
            print(f"⚠️ Skipping None frame at index {i}")
            continue
            
        # Ensure frame is the right size and type
        if frame.shape[:2] != (VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0]):
            frame = cv2.resize(frame, VIDEO_RESOLUTION)
            
        # Add crossfade effect at scene boundaries
        if i > 0 and i % frames_per_scene < transition_frames:
            # Crossfade with previous frame
            alpha = (i % frames_per_scene) / transition_frames
            if i > 0 and frames_list[i-1] is not None:
                prev_frame = frames_list[i-1]
                if prev_frame.shape[:2] == frame.shape[:2]:
                    blended = cv2.addWeighted(prev_frame, 1-alpha, frame, alpha, 0)
                    out.write(blended)
                    frames_written += 1
                else:
                    out.write(frame)
                    frames_written += 1
            else:
                out.write(frame)
                frames_written += 1
        else:
            out.write(frame)
            frames_written += 1
    
    out.release()
    
    # Verify video was created successfully
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path)
        print(f"✅ Video created: {video_path} ({file_size} bytes, {frames_written} frames)")
        if file_size < 1000:  # Very small file indicates error
            raise RuntimeError(f"Video file too small ({file_size} bytes), likely encoding error")
    else:
        raise RuntimeError("Video file was not created")
        
    return video_path

def generate_video_from_story(story, style='cinematic'):
    """Generate video with deep story understanding and consistent theming."""
    try:
        print(f"🎬 Starting timeline-based video generation for {style} style...")
        
        # 1. Extract Director's Cut scene breakdown
        print(f"🎬 Extracting cinematic scene breakdown...")
        director_analysis = extract_story_timeline(story, style)
        
        # Validate director analysis structure and type
        if not director_analysis or not isinstance(director_analysis, dict):
            print("⚠️ Director analysis failed or returned invalid format, using fallback")
            director_analysis = create_timeline_fallback(story, 5, style)
        
        # Extract components with type safety
        story_metadata = director_analysis.get('story_metadata', {})
        if not isinstance(story_metadata, dict):
            print("⚠️ Story metadata is not a dict, using fallback")
            director_analysis = create_timeline_fallback(story, 5, style)
            story_metadata = director_analysis.get('story_metadata', {})
            
        scenes = director_analysis.get('scenes', [])
        if not isinstance(scenes, list):
            print("⚠️ Scenes is not a list, using fallback")
            director_analysis = create_timeline_fallback(story, 5, style)
            scenes = director_analysis.get('scenes', [])
        
        
        # Extract metadata
        main_character = story_metadata.get('main_character', 'character')
        setting = story_metadata.get('setting', 'environment')
        
        print(f"🎬 Director's Cut: {len(scenes)} cinematic scenes")
        print(f"👤 Main character: {main_character}")
        print(f"🏠 Setting: {setting}")
        print(f"📋 Scene progression: {' -> '.join([s.get('scene_title', f'Scene {i+1}') for i, s in enumerate(scenes[:3])])}...")
        
        style_visual = VIDEO_STYLES.get(style, VIDEO_STYLES['cinematic'])['visual']
        visual_style = story_metadata.get('visual_style', style)
        
        print(f"🎨 Visual style: {visual_style} - {style_visual[:40]}...")
        print(f"⚡ Enhanced mode: 3 variations/scene, 25 FPS, 25 steps, 720x512 resolution")
        
        # 2. Load and prepare image generation
        # Use the returned pipe from load_models() instead of imported variable
        # (imported pipe is None at module load time)
        image_pipe = load_models()
        
        # Verify pipeline loaded successfully
        if image_pipe is None:
            raise RuntimeError("Failed to load StableDiffusion pipeline - pipe is None")
        
        # Disable safety checker to prevent black frames
        if hasattr(image_pipe, 'safety_checker'):
            image_pipe.safety_checker = None
            print("✅ Safety checker disabled for consistent generation")
        
        all_frames = []
        temp_images = []
        
        # STRONGLY enforced style-specific negative prompts to prevent style mixing
        style_negatives = {
            'cinematic': "cartoon, animated, anime, 3D render, cel-shaded, disney, pixar, illustration, drawing, sketch, toon, comic, animation, animated character, cartoon style, pixar style, disney style, 3d cartoon",
            'animated': "photorealistic, live action, realistic photography, film grain, documentary, real person, natural lighting, film camera, human skin texture, photo, photograph, real life, cinematic film, movie photography, film still, realistic render, lifelike, actual photo, candid photo, professional photography, childish cartoon, mickey mouse style, simple cartoon, kids cartoon, toddler cartoon, baby cartoon, simplistic art, crude drawing, stick figure, comic strip, newspaper comic, low detail cartoon, flat colors, chibi", 
            'realistic': "cartoon, animated, anime, 3D render, cel-shaded, disney, pixar, illustration, drawing, sketch, artistic, painted, toon, comic, animation, animated character, cartoon style, pixar style, disney style, 3d cartoon, painting, art style",
            'artistic': "photorealistic, live action, realistic photography, film grain, documentary, real person, natural lighting, film camera, cartoon, animated, photo, photograph, real life, cinematic film, movie photography, film still"
        }
        
        # Enhanced negative prompt to prevent distortion
        base_negative = "low quality, blurry, distorted, deformed, disfigured, ugly, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, disconnected limbs, mutation, mutated, bad proportions, duplicate, cropped, worst quality, jpeg artifacts, signature, watermark, username, text, error, malformed, gross proportions, cloned face, inconsistent style, mixed styles, style mixing"
        enhanced_negative = style_negatives.get(style, style_negatives['cinematic']) + ", " + base_negative
        
        # 3. Generate images from Director's visual descriptions
        total_images = len(scenes) * 3  # 3 variations per scene
        current_image = 0
        
        for scene_idx, scene in enumerate(scenes):
            # Ensure scene is a dictionary
            if not isinstance(scene, dict):
                print(f"⚠️ Scene {scene_idx} is not a dict, skipping")
                continue
                
            scene_title = scene.get('scene_title', f'Scene {scene_idx + 1}')
            visual_desc = scene.get('visual_description', 'scene action')
            mood = scene.get('mood', 'dramatic')
            
            print(f"🎬 Processing Scene {scene_idx + 1}/{len(scenes)}: {scene_title}")
            print(f"📸 Visual: {visual_desc[:60]}...")
            print(f"🎭 Mood: {mood}")
            
            # Create 3 variations of this scene
            scene_frames = []
            for var_idx in range(3):
                current_image += 1
                
                # Generate enhanced prompt using encoder-decoder architecture
                previous_scene = scenes[scene_idx - 1] if scene_idx > 0 else None
                image_prompt, enhanced_negative = create_enhanced_video_prompts(
                    scene, previous_scene, story_metadata, scene_idx, len(scenes), style
                )
                
                print(f"  🖼️  Generating enhanced image {current_image}/{total_images} ({scene_title}, Variation {var_idx + 1})...")
                print(f"  📝 Enhanced prompt: {image_prompt[:70]}...")
                
                # Enhanced generation parameters with semantic consistency
                import random
                
                # Character-stable seeding: same base seed per story so the
                # same character appearance is encouraged across all scenes.
                # Only the variation index (0-2) shifts it slightly.
                story_hash = abs(hash(story_metadata.get('main_character', '') +
                                      story_metadata.get('char_description', ''))) % 10000
                scene_seed = (story_hash + var_idx * 7) % 100000  # scene_idx NOT multiplied in — keeps character stable
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(scene_seed)
                
                # Style-specific settings — higher steps & guidance = better quality
                is_gpu = device == "cuda"
                if style == 'animated':
                    guidance_scale = 9.5   # high = follows prompt closely → consistent char
                    num_steps = 35 if is_gpu else 25
                elif style == 'cinematic':
                    guidance_scale = 8.5
                    num_steps = 35 if is_gpu else 25
                else:
                    guidance_scale = 8.0
                    num_steps = 30 if is_gpu else 22
                
                image = image_pipe(
                    image_prompt,
                    negative_prompt=enhanced_negative,
                    num_inference_steps=num_steps,  # Optimized for speed
                    guidance_scale=guidance_scale,
                    width=VIDEO_RESOLUTION[0],
                    height=VIDEO_RESOLUTION[1],
                    generator=generator,
                    eta=0.0,
                    num_images_per_prompt=1
                ).images[0]
                
                # Save and process
                temp_path = os.path.join(VIDEO_OUTPUT_DIR, f"temp_scene_{scene_idx}_{var_idx}.png")
                image.save(temp_path)
                temp_images.append(temp_path)
                
                # Apply appropriate motion (3 types for 3 variations)
                motion_types = ["static", "zoom_in", "zoom_out"]  # 3 motion types for 3 variations
                motion = motion_types[var_idx % len(motion_types)]
                
                frames = apply_animation_effects(temp_path, motion, IMAGE_DURATION)
                if frames:
                    scene_frames.extend(frames)
            
            all_frames.extend(scene_frames)
            
            # Clear GPU cache every 3 scenes to prevent memory errors
            if (scene_idx + 1) % 3 == 0:
                torch.cuda.empty_cache()
                print(f"  🧹 Cleared GPU cache after scene {scene_idx + 1}")
        
        # Final GPU cache clear
        torch.cuda.empty_cache()
        print("🧹 Final GPU cache clear")
        
        # 4. Create final video with smooth transitions
        print("🎞️  Creating final video from Director's Cut...")
        video_path = create_video_with_transitions(all_frames, len(scenes))
        
        # 5. Cleanup temporary files
        for temp_img in temp_images:
            try:
                os.remove(temp_img)
            except:
                pass
        
        print(f"✅ Timeline-based video generation complete: {video_path}")
        
        # Prepare response structure compatible with main.py
        story_analysis_for_return = {
            'scenes': director_analysis.get('scenes', []),
            'summary': f"Director's Cut video with {len(scenes)} cinematic scenes",
            'story_scenes': director_analysis.get('scenes', []),  # For backward compatibility
            'extracted_elements': {
                'main_story_theme': f"Cinematic scenes: {' -> '.join([s.get('scene_title', '') for s in scenes[:3]])}..."
            }
        }
        
        return {
            'success': True,
            'video_path': video_path,
            'total_scenes': len(scenes),
            'total_frames': len(all_frames),
            'story_analysis': story_analysis_for_return,
            'style': style
        }
        
    except Exception as e:
        print(f"❌ Error generating video: {e}")
        raise e