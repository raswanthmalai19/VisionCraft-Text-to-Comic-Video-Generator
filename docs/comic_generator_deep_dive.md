# comic_generator.py — Complete Deep Dive

---

## File Purpose
`comic_generator.py` is the **core AI engine** for the comic generation feature. It handles everything from reading the raw story text to producing assembled multi-page comic PNG files.

---

## Module-Level Imports & Availability Flags

```python
import torch                  # PyTorch — GPU/CPU tensor computation
import os, base64, re, math   # Standard library
from io import BytesIO        # In-memory byte buffer for base64 encoding
import numpy as np            # Numerical arrays
import warnings               # Suppress deprecation warnings
```

### Conditional Imports (Graceful Degradation)
```python
# ── Stable Diffusion ──────────────────────────────────────────────
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except:
    DIFFUSERS_AVAILABLE = False   # Comic generation will be disabled

# ── PIL ───────────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except:
    PIL_AVAILABLE = False

# ── NLTK ──────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.tree import Tree
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# ── Transformers / ML ─────────────────────────────────────────────
try:
    from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False
```

Each `AVAILABLE` flag is exported and imported by `video_generator.py` and `main.py` to decide which code paths to use.

---

## Configuration Constants

```python
MODEL_ID = "runwayml/stable-diffusion-v1-5"   # The SD model to use
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = "output"         # Where panel PNGs are saved
PANELS_PER_ROW = 2            # Default layout (overridden later)
IMAGE_SIZE = (512, 512)       # Each panel size in pixels
FONT_SIZE = 20                # Caption font size
MARGIN = 20                   # Text margin from image edge

# Panel count bounds
MIN_PANELS = 2                # Story won't be broken into fewer than 2
MAX_PANELS = 20               # Hard cap — avoids memory exhaustion
OPTIMAL_WORDS_PER_PANEL = 15  # Target word count per panel

# Multi-page layout
MIN_PANELS_PER_PAGE = 4       # Don't make pages with < 4 panels
MAX_PANELS_PER_PAGE = 8       # Don't pack more than 8 per page
PREFERRED_PANELS_PER_PAGE = 6 # Try to target 6 panels/page
```

---

## Comic Themes

8 pre-defined themes, each with a name and a **Stable Diffusion style string**:

```python
COMIC_THEMES = {
    'classic': {
        'name': 'Classic Comic Book',
        'style': 'classic American comic book style, clean line art, vibrant primary colors, 
                  professional comic illustration, superhero comic aesthetic'
    },
    'manga': {
        'name': 'Manga/Anime Style',
        'style': 'Japanese manga style, detailed line art, expressive large eyes, 
                  dynamic action poses, black and white with screentone effects'
    },
    'superhero': { ... },
    'cartoon':   { ... },
    'noir':      { ... },
    'fantasy':   { ... },
    'scifi':     { ... },
    'webcomic':  { ... }
}
```

The `style` string is injected directly into the Stable Diffusion prompt to condition image style.

---

## Lazy Model Loading: `load_models()`

```python
# Global state
pipe = None              # Stable Diffusion pipeline
sentence_model = None    # SentenceTransformer
summarization_model = None  # BART

def load_models():
    global pipe, sentence_model, summarization_model
    
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,           # Disabled to avoid black images
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        if device == "cuda":
            pipe.enable_attention_slicing(1)  # Reduces peak VRAM
            pipe.enable_vae_slicing()          # Saves more VRAM
    
    if sentence_model is None:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if summarization_model is None:
        summarization_model = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=0 if DEVICE == "cuda" else -1   # 0=GPU, -1=CPU
        )
    
    return pipe
```

**Why `safety_checker=None`?**  
The SD safety checker can produce entirely black images when it falsely detects NSFW content in fantasy/action scenes. Disabling it prevents this.

---

## TextEncoder Class

```python
class TextEncoder:
    def encode_sentences(self, sentences):
        global sentence_model
        
        # Priority 1: SentenceTransformer neural embedding
        if sentence_model is not None:
            return sentence_model.encode(sentences)
            # Returns: numpy array shape (N, 384)
        
        # Priority 2: TF-IDF sparse to dense
        if len(sentences) > 1:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            return vectorizer.fit_transform(sentences).toarray()
        
        # Priority 3: Basic word count vectors (last resort)
        return self._basic_encode_sentences(sentences)
    
    def _basic_encode_sentences(self, sentences):
        # Build a vocabulary from all unique words
        # Each sentence becomes a word-count vector
        # Very rough approximation of semantic similarity
```

---

## SemanticChunker Class

The most complex class in the file. See [ai_models_and_concepts.md](ai_models_and_concepts.md#7-semanticchunker-custom-class) for the conceptual explanation.

### `_analyze_sentences(sentences)` — Sentence Metadata
For each sentence, produces a dict:
```python
{
    'text': "Luna discovered the key.",
    'word_count': 4,
    'entities': {
        'people': ['Luna'],
        'places': [],
        'things': ['key']
    },
    'sentence_type': 'action',   # 'action'|'dialogue'|'description'|'transition'|'narrative'
    'importance_score': 4        # higher = more visually interesting
}
```

### `_classify_sentence_type(sentence)` — Rule-Based Classification
```python
action_keywords   = ['ran', 'jumped', 'hit', 'threw', 'grabbed', ...]  → 'action'
'"' or '!' or '?'                                                        → 'dialogue'
description words = ['was', 'were', 'had', 'stood', 'appeared', ...]   → 'description'
transition words  = ['then', 'next', 'after', 'suddenly', 'finally']   → 'transition'
everything else                                                          → 'narrative'
```

### `_calculate_importance(sentence)` — Scoring
```python
score = 0
if 5 <= word_count <= 15:    score += 2   # Ideal length for a panel
if visual word found:        score += 1   # "see", "bright", "dark", etc.
if action word found:        score += 2   # "fight", "run", "fly" — very visual
if emotion word found:       score += 1   # "happy", "scared", etc.
```

### `_group_by_semantics(analyzed_sentences)` — Core Grouping
```python
similarity_matrix = cosine_similarity(embeddings)

for each sentence i:
    group = [sentence_i]
    for each sentence j after i:
        if similarity_matrix[i][j] > 0.3:   # semantic similarity threshold
            group.append(sentence_j)
        if len(group) >= 3:
            break  # Don't make groups too large
    groups.append(group)
```

### `_create_narrative_chunks(groups)` — Output Format
Each group becomes a chunk dict with combined text, importance score, and merged entity lists.

---

## Story Analysis Functions

### `smart_sentence_split(text)`
```python
if NLTK_AVAILABLE:
    sentences = sent_tokenize(text)   # NLTK punkt model
else:
    sentences = re.split(r'[.!?]+', text)   # regex fallback

# Filter: remove blanks and very short fragments (< 3 chars)
sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
```

### `analyze_story_structure(text)` — Fallback (no Transformers)
Classic approach without neural models:
1. Count total words.
2. Calculate target panel count: `ceil(total_words / 15)` capped to [2, 20].
3. Group sentences evenly into that many panels.

### `advanced_story_analysis(text)` — Main Path (with Transformers)
1. Uses `SemanticChunker.extract_semantic_units(text)` for semantically meaningful chunks.
2. Splits any chunk with > 25 words into two.
3. Merges smallest chunks to reach `MIN_PANELS` (2).
4. Combines shortest adjacent chunks to stay at `MAX_PANELS` (20).

### `group_sentences_into_panels(sentences, target_panels)`
Assigns sentences to panels while maintaining narrative flow:
- Calculates `sentences_per_panel = len(sentences) / target_panels`
- As it processes each sentence, tracks how many to assign to current panel.
- Adjusts the target dynamically for remaining panels.

---

## Character & Setting Extraction

### `extract_characters_and_setting(story)` — Full Logic

**Character extraction:**
1. Tokenize every sentence.
2. For each word: skip if not capitalized, in STOP_WORDS, or in PLACE_WORDS.
3. Skip if it follows "the/a/an" (article → likely a place name, not a person).
4. Skip if the next word is in PLACE_WORDS (e.g., "Whispering **Woods**").
5. Count remaining words by frequency → top 3 = main characters.

**The PLACE_WORDS set (prevents false positives):**
```python
PLACE_WORDS = {
    'woods', 'forest', 'castle', 'fortress', 'tower', 'ocean', 'sea',
    'mountain', 'village', 'realm', 'kingdom', 'whispering', 'obsidian',
    'ancient', 'enchanted', 'haunted', 'forbidden', 'dark', 'golden', ...
}
```

**Setting extraction:**
Simple keyword matching against 47 setting indicators:
```python
setting_indicators = ['castle', 'forest', 'city', 'school', 'house', 'mountain',
                       'beach', 'space', 'hospital', 'laboratory', 'garden', ...]
```
Words found → sorted by frequency → top 2 returned.

---

## Prompt Engineering Functions

### `create_enhanced_prompt_v2(panel_text, theme_style, characters, setting, ...)`
**Steps:**
1. `extract_visual_elements(panel_text)` → lighting, colors, action poses
2. `enhance_panel_description_v2(panel_text, visual_elements)` → replace abstract words
3. Build style tag from theme name: `"manga comic panel"`, `"superhero comic panel"`, etc.
4. Assemble: `[enhanced_text, style, "character_name character", "location background", visual_elements, quality]`
5. Build negative prompt: `"blurry, low quality, text, watermark, bad anatomy"`

### `extract_visual_elements(text)`
Rule-based extraction of visual cues from text:
```python
# Lighting detection (first match wins)
{'dark': 'dark atmosphere', 'bright': 'bright lighting', 'sun': 'sunlight', ...}

# Color detection
['red', 'blue', 'green', 'yellow', 'black', 'white', 'gold', 'silver']

# Action → visual pose
{'fight': 'action scene', 'run': 'running motion', 'fly': 'flying pose', ...}

# Emotion → expression
{'happy': 'happy expression', 'sad': 'sad expression', ...}
```

### `enhance_panel_description_v2(panel_text, visual_elements)`
Replaces vague common phrases with visually specific ones:
```python
replacements = {
    'was happy':   → 'smiled joyfully',
    'was sad':     → 'looked downcast',
    'was angry':   → 'glared angrily',
    'was scared':  → 'trembled in fear',
    'discovered':  → 'found and examined',
    'realized':    → 'suddenly understood',
    ' said':       → ' declared',
    ' told':       → ' announced',
}
```

---

## Layout Functions

### `calculate_optimal_layout(num_panels)` → `(cols, rows)`
```
2 panels  → 2×1
3–4 panels → 2×2
5–6 panels → 3×2
7–9 panels → 3×3
10+ panels → 4 × ceil(N/4)
```

### `calculate_page_layout(total_panels)` → `[(panels, (cols, rows)), ...]`
Distributes panels across multiple pages:
- If total ≤ 8: single page
- Otherwise: create pages of 6 panels each
- Last page: adjust if it would have < 4 panels (steal from previous page)

---

## Main Generation Function: `generate_comic(story, theme)`

```python
def generate_comic(story, theme='classic'):
    load_models()             # step 1: ensure all models loaded
    
    # step 2: semantic analysis
    if TRANSFORMERS_AVAILABLE and sentence_model is not None:
        panels = advanced_story_analysis(story)   # NLP-powered
    else:
        panels = analyze_story_structure(story)   # regex fallback
    
    # step 3: layout calculation
    page_layouts = calculate_page_layout(len(panels))
    
    # step 4: character/setting extraction
    characters, setting = extract_characters_and_setting(story)
    story_context = analyze_story_context(story)
    
    # step 5: generate one image per panel
    for i, panel_text in enumerate(panels):
        prompt, neg_prompt = create_enhanced_prompt_v2(
            panel_text, theme_style, characters, setting, i+1, num_panels
        )
        
        seed = (story_hash + i * 137) % 100000
        generator = torch.Generator(device).manual_seed(seed)
        
        image = pipe(
            prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=25,   # 25 on GPU, 30 on CPU
            guidance_scale=7.5,
            width=512, height=512,
            generator=generator,
            eta=0.0
        ).images[0]
        
        # step 6: add caption text overlay
        draw = ImageDraw.Draw(image)
        wrapped_text = textwrap.fill(panel_text, width=45)
        # draw white box behind text, then draw text in black
        
        image.save(f"output/semantic_panel_{i+1}.png")
        generated_images.append(image)
    
    # step 7: assemble pages
    for page_num, (panels_on_page, (cols, rows)) in enumerate(page_layouts):
        comic = Image.new("RGB", (cols * 512, rows * 512), "white")
        for local_idx in range(panels_on_page):
            img = generated_images[panel_idx]
            x_pos = (local_idx % cols) * 512
            y_pos = (local_idx // cols) * 512
            comic.paste(img, (x_pos, y_pos))
            # draw black border
        
        comic.save(f"output/semantic_comic_page_{page_num}.png")
        imgs_base64.append(base64.b64encode(comic.tobytes()))
    
    return comic_paths, imgs_base64, num_panels, layout_info
```

---

## Helper Functions Quick Reference

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `smart_sentence_split(text)` | story string | list of sentences | NLTK or regex splitting |
| `count_words(text)` | string | int | NLTK or split() word count |
| `analyze_story_context(story)` | story string | `{theme, mood, genre}` | Keyword-based genre/mood detection |
| `enhanced_text_analysis_demo(story)` | story string | analysis dict | Demo mode — shows NLP results |
| `run_cli()` | — | — | CLI entry point |
