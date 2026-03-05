# VisionCraft — Text to Comic & Video Generator
### Complete Project Documentation Index

---

## What is VisionCraft?

**VisionCraft** is an AI-powered web application that takes a plain English story (a paragraph or more) and transforms it into:

1. **Multi-page Comic Strips** — Panel-by-panel illustrations in 8 different art styles, generated using Stable Diffusion.
2. **Animated Story Videos** — Cinematic scene-by-scene videos with motion effects, generated from the same story.

The entire pipeline runs **locally on your machine** — no external API calls for generation. It uses state-of-the-art open-source NLP and image generation models.

---

## Documentation Files

| File | What It Covers |
|------|---------------|
| [README.md](README.md) | This file — project overview and index |
| [project_structure.md](project_structure.md) | Every folder and file explained |
| [architecture.md](architecture.md) | Full system architecture and data flow |
| [comic_generator_deep_dive.md](comic_generator_deep_dive.md) | Line-by-line breakdown of `comic_generator.py` |
| [video_generator_deep_dive.md](video_generator_deep_dive.md) | Line-by-line breakdown of `video_generator.py` |
| [ai_models_and_concepts.md](ai_models_and_concepts.md) | Every AI model, NLP technique, and concept used |
| [web_interface.md](web_interface.md) | Frontend, Flask routes, and API contract |

---

## Quick Start

```bash
# 1. Activate the virtual environment
source svd_env/Scripts/activate    # Windows
# or
source svd_env/bin/activate         # macOS/Linux

# 2. Start the web server
python main.py

# 3. Open browser
http://localhost:5000
```

---

## Core Technology Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Flask (Python) |
| Image Generation | Stable Diffusion v1.5 (runwayml) via Diffusers |
| Text Understanding | NLTK, Sentence-Transformers, BART (facebook/bart-large-cnn) |
| Semantic Similarity | all-MiniLM-L6-v2 + TF-IDF + cosine similarity |
| Image Processing | Pillow (PIL) |
| Video Assembly | OpenCV (cv2) + MoviePy |
| NLP | NLTK (NER, POS tagging, tokenization) |
| Deep Learning Runtime | PyTorch |

---

## How It Works (30-second summary)

```
Your Story (text)
      │
      ▼
[NLP Pipeline] ──→ split into sentences → semantic chunking → panel/scene list
      │
      ▼
[Prompt Engineering] ──→ each panel/scene becomes a Stable Diffusion prompt
      │
      ▼
[Stable Diffusion v1.5] ──→ generates one 512×512 image per panel
      │
      ▼
[Post-processing]
  Comic: PIL combines panels into a grid layout → PNG pages
  Video: OpenCV adds Ken Burns effects → MP4 video
      │
      ▼
[Flask API] ──→ base64 images / video file path sent to browser
      │
      ▼
[Frontend (index.html)] ──→ displays comic pages or plays video
```
