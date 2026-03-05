# Project Structure — Every Folder & File Explained

---

## Top-Level Layout

```
text to comic - Copy/
│
├── main.py                  ← Flask web server + all API routes
├── comic_generator.py       ← Comic generation engine (NLP + Stable Diffusion)
├── video_generator.py       ← Video generation engine (scenes + OpenCV)
│
├── templates/
│   └── index.html           ← Single-page frontend (HTML + CSS + JS)
│
├── output/                  ← Generated comic PNG files saved here
├── video_output/            ← Generated MP4 videos saved here
│
├── svd_env/                 ← Python virtual environment
│   ├── pyvenv.cfg
│   ├── Lib/site-packages/   ← All installed Python libraries
│   └── Scripts/             ← activate/deactivate scripts
│
└── docs/                    ← This documentation folder
    ├── README.md
    ├── project_structure.md  ← YOU ARE HERE
    ├── architecture.md
    ├── comic_generator_deep_dive.md
    ├── video_generator_deep_dive.md
    ├── ai_models_and_concepts.md
    └── web_interface.md
```

---

## File-by-File Breakdown

---

### `main.py` — The Web Server
**Role:** Entry point. Starts Flask, defines all HTTP routes, wires the comic and video modules together.

**Key things it does:**
- Imports `comic_generator` and `video_generator` with graceful fallback (if one fails, the other still works).
- Defines `GET /` → serves `index.html`.
- Defines `POST /generate` → calls `generate_comic(story, theme)` and returns JSON with base64 PNG pages.
- Defines `POST /generate_video` → calls `generate_video_from_story(story, style)` and returns the video file path.
- Defines `GET /output/<filename>` and `GET /video_output/<filename>` → static file serving for generated content.
- Validates the video file with OpenCV before serving (checks frame count, FPS, duration).
- Sets proper HTTP headers for video streaming (`Content-Type: video/mp4`, `Accept-Ranges: bytes`).
- Can also run in CLI mode: `python main.py cli`.

**Imports used:**
```python
from flask import Flask, render_template, request, jsonify, send_from_directory
from comic_generator import generate_comic, COMIC_THEMES, OUTPUT_DIR, DIFFUSERS_AVAILABLE
from video_generator import generate_video_from_story, VIDEO_OUTPUT_DIR
```

---

### `comic_generator.py` — Comic Engine
**Role:** The brain of comic generation. Does NLP, prompt engineering, image generation, and layout.

**Size:** ~1,400+ lines  
**Responsibilities:**
1. **Text Analysis** — Splits story into semantically meaningful panels using NLTK + Sentence Transformers.
2. **Semantic Chunking** — Groups related sentences using cosine similarity on sentence embeddings.
3. **Prompt Engineering** — Converts each panel into a rich Stable Diffusion text prompt.
4. **Image Generation** — Runs Stable Diffusion v1.5 to render each panel.
5. **Layout** — Arranges panels into multi-page grid layouts using PIL.
6. **Export** — Saves PNG files and returns base64 strings to Flask.

**Key classes and functions:**

| Name | Type | Purpose |
|------|------|---------|
| `TextEncoder` | Class | Encodes sentences to vectors (sentence-transformers or TF-IDF fallback) |
| `SemanticChunker` | Class | Groups sentences into semantically coherent chunks |
| `load_models()` | Function | Lazy-loads Stable Diffusion + NLP models on first request |
| `generate_comic()` | Function | Main entry point — orchestrates the full pipeline |
| `smart_sentence_split()` | Function | NLTK or regex sentence splitter |
| `extract_characters_and_setting()` | Function | NER-based character/setting detection |
| `create_enhanced_prompt_v2()` | Function | Builds the Stable Diffusion prompt for each panel |
| `calculate_page_layout()` | Function | Decides how many panels per page for multi-page comics |
| `analyze_story_context()` | Function | Detects genre (fantasy/sci-fi/mystery) and mood |
| `advanced_story_analysis()` | Function | Full semantic pipeline using `SemanticChunker` |

---

### `video_generator.py` — Video Engine
**Role:** Generates MP4 videos from stories. Reuses the comic pipeline for image generation but adds motion effects and video assembly.

**Size:** ~1,500+ lines  
**Responsibilities:**
1. **Scene Extraction** — Divides the story into cinematic scenes (not comic panels).
2. **Character Anchor** — Builds a fixed visual description of the main character for consistency across frames.
3. **Prompt Engineering** — Creates scene-specific Stable Diffusion prompts with stronger visual anchoring.
4. **Image Generation** — Generates 3 image variations per scene using Stable Diffusion.
5. **Motion Effects** — Applies Ken Burns effects (zoom in/out, pan left/right) using OpenCV.
6. **Video Assembly** — Stitches frames with crossfade transitions using OpenCV VideoWriter.

**Key functions:**

| Name | Purpose |
|------|---------|
| `extract_story_timeline()` | Top-level scene extractor — uses transformer pipeline or fallback |
| `extract_scenes_with_transformers()` | Advanced scene extraction using SemanticChunker + BART |
| `build_character_description()` | Creates a consistent character anchor string |
| `extract_visual_description()` | NLP-based scene-specific visual description |
| `create_enhanced_video_prompts()` | Builds SD prompts with char anchor + scene specifics |
| `apply_animation_effects()` | Ken Burns zoom/pan effects per image |
| `create_video_with_transitions()` | OpenCV VideoWriter + crossfade assembly |
| `generate_video_from_story()` | Main entry point — full video pipeline |

---

### `templates/index.html` — The Frontend
**Role:** Single-page web UI built with vanilla HTML/CSS/JavaScript.

**Sections:**
- **Header** — "VisionCraft" title and subtitle.
- **Story Input** — A `<textarea>` for the user's story text.
- **Theme Selector** — 8 comic art style choices (dropdown).
- **Generate Comic Button** — Triggers `POST /generate`.
- **Video Section** — Style selector (4 cinematic styles) + "Generate Video" button.
- **Loading Spinner** — Shown during generation.
- **Result Area** — Dynamically rendered with generated comic pages or video player.
- **Example Stories** — 3 clickable examples to prefill the textarea.

**JavaScript functions:**
- `generateComic()` — Async fetch to `/generate`, renders base64 images.
- `generateVideo()` — Async fetch to `/generate_video`, renders `<video>` tag.
- `downloadImage(base64, filename)` — Downloads a single comic page.
- `downloadAllPages(images)` — Staggered download of all comic pages.
- `updateThemePreview()` — Updates description text when theme changes.
- `testVideoUrl(url)` — Debug: checks if video URL is accessible.

---

### `output/` — Comic Output Directory
- Created automatically by `comic_generator.py`.
- Stores individual panel images: `semantic_panel_1.png`, `semantic_panel_2.png`, etc.
- Stores assembled comic pages: `semantic_comic_page_1.png`, `semantic_comic_page_2.png`, etc.
- Served by Flask at `GET /output/<filename>`.

---

### `video_output/` — Video Output Directory
- Created automatically by `video_generator.py`.
- Stores temporary per-scene images: `temp_scene_0_0.png`, `temp_scene_0_1.png`, etc. (deleted after assembly).
- Stores the final MP4: `story_video_YYYYMMDD_HHMMSS.mp4`.
- Served by Flask at `GET /video_output/<filename>`.

---

### `svd_env/` — Python Virtual Environment
A self-contained Python environment with all dependencies pre-installed.

**Key installed packages (in `Lib/site-packages/`):**

| Package | Purpose |
|---------|---------|
| `diffusers` | Hugging Face Diffusers — runs Stable Diffusion |
| `torch` | PyTorch — tensor operations, GPU support |
| `transformers` | Hugging Face Transformers — BART, tokenizers |
| `sentence_transformers` | Sentence embeddings (`all-MiniLM-L6-v2`) |
| `nltk` | Natural language processing (tokenization, NER, POS tagging) |
| `PIL` / `Pillow` | Image creation, drawing, font rendering |
| `cv2` / `opencv-python` | Video frame manipulation, VideoWriter |
| `moviepy` | Optional higher-level video editing |
| `flask` | Web server and HTTP routing |
| `sklearn` | TF-IDF vectorizer, cosine similarity |
| `numpy` | Array/matrix operations |
| `google-generativeai` | (Installed but not actively used — legacy from earlier version) |
| `accelerate` | Hugging Face Accelerate — model optimization |
| `huggingface_hub` | HF model downloading and caching |

**Note:** The `.whl` files for numpy and scipy in site-packages are Windows-specific wheels — this project was originally developed on Windows.

---

### `__pycache__/`
Python's bytecode cache. Auto-generated. Can be safely deleted and will be recreated on next run.
