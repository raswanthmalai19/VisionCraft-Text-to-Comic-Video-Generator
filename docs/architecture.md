# System Architecture & Data Flow

---

## Overview

VisionCraft is a **single-server, multi-pipeline application** where both the web serving and AI computation run in the same Python process. When a request comes in, it triggers a heavy AI pipeline that may take 5–15 minutes.

---

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                         index.html                              │
│   ┌───────────────────────────────────────────────────────┐     │
│   │  textarea (story)  │  theme/style picker  │  buttons  │     │
│   └──────────────────────────┬────────────────────────────┘     │
└─────────────────────────────┼───────────────────────────────────┘
                               │ HTTP POST (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK SERVER (main.py)                        │
│                                                                  │
│  POST /generate ─────────────────────────────────────────────┐  │
│  POST /generate_video ────────────────────────────────────┐  │  │
│  GET  /output/<file>        GET /video_output/<file>      │  │  │
└────────────────────────────────────────────────────────────┼──┼──┘
                                                             │  │
                    ┌────────────────────────┐               │  │
                    │   comic_generator.py   │◄──────────────┘  │
                    │   NLP + SD pipeline    │                  │
                    └────────────┬───────────┘                  │
                                 │                              │
                    ┌────────────▼───────────┐                  │
                    │   video_generator.py   │◄─────────────────┘
                    │  Scene + Video pipe    │
                    │  (reuses comic funcs)  │
                    └────────────────────────┘
```

---

## Comic Generation Pipeline (Detailed)

```
story_text (string)
      │
      ▼
┌─────────────────────────────────────────────┐
│  1. TEXT ANALYSIS                            │
│                                             │
│  smart_sentence_split()                     │
│    ├─ NLTK sent_tokenize (if available)     │
│    └─ regex split on [.!?] (fallback)       │
│                                             │
│  analyze_story_context()                    │
│    ├─ detects genre: fantasy/sci-fi/mystery │
│    └─ detects mood: dark/cheerful/exciting  │
│                                             │
│  extract_characters_and_setting()           │
│    ├─ NLTK NER (ne_chunk)                   │
│    ├─ capitalized word frequency counting  │
│    └─ setting keyword matching              │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  2. SEMANTIC CHUNKING (SemanticChunker)     │
│                                             │
│  TextEncoder.encode_sentences()             │
│    ├─ SentenceTransformer all-MiniLM-L6-v2  │
│    └─ TF-IDF vectorizer (fallback)          │
│                                             │
│  cosine_similarity(embeddings)              │
│    └─ group sentences with similarity > 0.3 │
│                                             │
│  Each group → one "chunk" (future panel)    │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  3. PANEL SIZING & MULTI-PAGE LAYOUT        │
│                                             │
│  advanced_story_analysis()                  │
│    ├─ split oversized chunks (>25 words)    │
│    ├─ merge undersized to reach MIN_PANELS  │
│    └─ cap at MAX_PANELS (20)               │
│                                             │
│  calculate_page_layout()                    │
│    ├─ up to 8 panels/page                  │
│    ├─ preferred: 6 panels/page             │
│    └─ returns grid dims (cols × rows)       │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  4. PROMPT ENGINEERING (per panel)          │
│                                             │
│  create_enhanced_prompt_v2()                │
│    ├─ enhance_panel_description_v2()        │
│    │   └─ replaces abstract words with      │
│    │      visual equivalents                │
│    ├─ extract_visual_elements()             │
│    │   └─ lighting, colors, actions         │
│    ├─ injects: character name + location    │
│    ├─ injects: theme style string           │
│    └─ builds negative_prompt               │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  5. IMAGE GENERATION (per panel)            │
│                                             │
│  Stable Diffusion v1.5                      │
│    ├─ model: runwayml/stable-diffusion-v1-5 │
│    ├─ 25 steps (GPU) / 30 steps (CPU)       │
│    ├─ guidance_scale = 7.5                  │
│    ├─ 512×512 px                            │
│    ├─ deterministic seed per panel          │
│    └─ outputs: PIL Image                    │
│                                             │
│  PIL post-processing                        │
│    ├─ draw caption text on image            │
│    ├─ white background box behind text      │
│    └─ save to output/semantic_panel_N.png   │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  6. PAGE ASSEMBLY                           │
│                                             │
│  PIL Image.new("RGB", (width, height))      │
│    ├─ paste each panel at grid position     │
│    ├─ draw black border around each panel   │
│    └─ save as semantic_comic_page_N.png     │
│                                             │
│  base64 encode each page PNG                │
│  → return to Flask → JSON response          │
└─────────────────────────────────────────────┘
```

---

## Video Generation Pipeline (Detailed)

```
story_text (string), style (string)
      │
      ▼
┌─────────────────────────────────────────────┐
│  1. SCENE EXTRACTION                        │
│                                             │
│  extract_story_timeline()                   │
│    └─ decides target_scenes count           │
│       (3 scenes for <30 words,              │
│        up to 15 for long stories)           │
│                                             │
│  extract_scenes_with_transformers()         │
│    ├─ SemanticChunker.extract_semantic_units│
│    ├─ distribute chunks → N scenes          │
│    ├─ generate_scene_description() per scene│
│    ├─ extract_mood() per scene              │
│    ├─ extract_key_elements() per scene      │
│    └─ generate_scene_title() per scene      │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  2. CHARACTER ANCHOR                        │
│                                             │
│  build_character_description()              │
│    ├─ detects gender (pronoun counting)     │
│    ├─ detects age (keywords: boy/old/etc.)  │
│    ├─ detects hair color                    │
│    ├─ detects clothing                      │
│    └─ detects story-specific props          │
│       e.g. "adult man, dark hair, coat,     │
│             holding a brass key"            │
│                                             │
│  This SAME string is injected into          │
│  every scene prompt to keep the             │
│  character visually consistent.             │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  3. PROMPT ENGINEERING (per scene × 3 vars) │
│                                             │
│  create_enhanced_video_prompts()            │
│    ├─ char_anchor (fixed)                  │
│    ├─ scene_specific actions/objects        │
│    ├─ setting_phrase (safe phrasing)        │
│    │   (water settings → "coastal shore")  │
│    ├─ mood atmosphere                       │
│    ├─ style keywords                        │
│    ├─ progressive camera:                   │
│    │   first 15% → "establishing wide shot"│
│    │   last 15% → "dramatic close-up"       │
│    │   middle → "medium narrative shot"     │
│    └─ comprehensive negative_prompt         │
│       (prevents creatures, style mixing)    │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  4. IMAGE GENERATION (scene × 3 variations) │
│                                             │
│  Stable Diffusion v1.5 (720×512 px)         │
│    ├─ animated style: guidance=9.5, 35 steps│
│    ├─ cinematic style: guidance=8.5, 35 steps│
│    ├─ character-stable seeding              │
│    │   (seed based on char description,     │
│    │    not scene index — keeps char same)  │
│    └─ 3 images per scene                    │
│       Variation 0 → static                 │
│       Variation 1 → zoom_in                 │
│       Variation 2 → zoom_out               │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  5. MOTION EFFECTS (Ken Burns)              │
│                                             │
│  apply_animation_effects()                  │
│    ├─ resize image to 130% of target (1.3×) │
│    ├─ zoom_in: crop shrinks over time       │
│    ├─ zoom_out: crop grows over time        │
│    ├─ pan_right/left: offset slides         │
│    ├─ 25 FPS, 2 seconds per image           │
│    └─ returns list of numpy frames          │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│  6. VIDEO ASSEMBLY                          │
│                                             │
│  create_video_with_transitions()            │
│    ├─ OpenCV VideoWriter                    │
│    │   codec: mp4v → XVID → MJPG (fallback) │
│    ├─ 720×512 @ 25 FPS                      │
│    ├─ crossfade at scene boundaries         │
│    │   (alpha blend prev/next frame)        │
│    ├─ verify file size > 1000 bytes         │
│    └─ save as story_video_TIMESTAMP.mp4    │
└─────────────────────────────────────────────┘
```

---

## Model Loading Strategy (Lazy Loading)

Models are NOT loaded at server startup. They are loaded the **first time** a generation request arrives:

```python
# Global variables, initially None
pipe = None              # Stable Diffusion pipeline
sentence_model = None    # SentenceTransformer
summarization_model = None  # BART

def load_models():
    global pipe, sentence_model, summarization_model
    if pipe is None:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, ...)
    if sentence_model is None:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    if summarization_model is None:
        summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
    return pipe
```

**Why lazy loading?**  
To avoid slowing down the server startup. Model download/load can take 1–5 minutes the first time.

---

## GPU vs CPU Behavior

| Aspect | GPU (CUDA) | CPU |
|--------|-----------|-----|
| SD dtype | `torch.float16` (half precision) | `torch.float32` (full precision) |
| Inference steps | 25 (comic), 35 (video) | 30 (comic), 25 (video) |
| Speed | ~2–5 minutes total | ~30–90 minutes total |
| Memory optimizations | attention slicing, VAE slicing | None needed |
| Typical VRAM needed | ~4 GB (float16) | N/A |

---

## Dependency Fallback Chain

The system degrades gracefully if packages are missing:

```
Sentence Transformers available?
  YES → use all-MiniLM-L6-v2 embedding for semantic chunking
  NO  → try TF-IDF vectorizer
    TF-IDF available?
      YES → use TF-IDF cosine similarity
      NO  → use basic word count vectors

NLTK available?
  YES → use sent_tokenize, pos_tag, ne_chunk
  NO  → use regex sentence splitting

BART (summarization) available?
  YES → use BART to summarize each scene
  NO  → use extract_visual_description() (NLP-based fallback)

Diffusers available?
  YES → generate images with Stable Diffusion
  NO  → raise RuntimeError (cannot generate without SD)
```
