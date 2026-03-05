# VisionCraft: Text-to-Comic-Video-Generator

Transform your stories into stunning visual comic strips and animated videos using AI.

## Overview

**VisionCraft** is an intelligent storytelling tool that converts text narratives into:
- **Comic strips** — Multi-panel manga-style illustrations
- **Animated videos** — Cinematic scenes with Ken Burns motion effects

Powered by Stable Diffusion v1.5, advanced NLP, and semantic understanding.

## Features

✨ **AI-Powered Image Generation** — Uses Stable Diffusion for photorealistic and stylized art  
📖 **Smart Story Parsing** — Semantic chunking groups related sentences into cohesive scenes  
🎬 **Video Animation** — Ken Burns effects with smooth crossfades between scenes  
🎨 **Multiple Art Styles** — Manga, realistic, anime, comic art modes  
🚀 **GPU & CPU Support** — Automatic detection and fallback optimization  
⚡ **Lazy Model Loading** — Models load on-demand to minimize startup time  

## Project Structure

```
.
├── main.py                      # Flask web server & REST API
├── comic_generator.py           # NLP pipeline & SD image generation for comics
├── video_generator.py           # Scene extraction & video assembly with Ken Burns
├── templates/
│   └── index.html              # Single-page frontend interface
├── output/                      # Generated comic PNGs (gitignored)
├── video_output/               # Generated MP4s (gitignored)
├── svd_env/                    # Python virtual environment (gitignored)
├── docs/                       # Comprehensive documentation
│   ├── README.md              # Project overview
│   ├── project_structure.md   # Every file & folder explained
│   ├── architecture.md        # System design & data flow
│   ├── ai_models_and_concepts.md  # 15 AI/ML deep dives
│   ├── comic_generator_deep_dive.md
│   ├── video_generator_deep_dive.md
│   ├── web_interface.md       # API reference
│   └── study_guide.md         # Recommended reading order
└── .gitignore
```

## Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ VRAM recommended (GPU) or 16GB+ RAM (CPU)
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/raswanthmalai19/VisionCraft-Text-to-Comic-Video-Generator.git
   cd VisionCraft-Text-to-Comic-Video-Generator
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv svd_env
   source svd_env/bin/activate  # On Windows: svd_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install flask torch torchvision diffusers transformers sentence-transformers nltk pillow opencv-python numpy scikit-learn
   ```

4. **Run the server**
   ```bash
   python main.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

### Usage

**Via Web UI (Recommended):**
- Enter a story (or use examples)
- Select art style/theme
- Click "Generate Comic" or "Generate Video"
- Download the output

**Via CLI:**
```bash
python main.py cli
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Image Generation** | Stable Diffusion v1.5 | Core AI model for text-to-image |
| **Text Understanding** | NLTK, Sentence-Transformers | NLP & semantic similarity |
| **Summarization** | BART Large CNN | Scene description generation |
| **Web Framework** | Flask | REST API & static content serving |
| **Video Processing** | OpenCV | Frame assembly & Ken Burns effects |
| **Deep Learning** | PyTorch | Tensor computation & GPU acceleration |

## Key Concepts

### Semantic Chunking
Stories are broken into panels by finding semantically similar sentences (using cosine similarity on embeddings). Sentences about the same action/scene are grouped together.

### Character Consistency (Video)
A "character anchor" string (e.g., "adult male, brown hair, blue shirt") is injected into every video frame prompt. The SD seed is based on character description, not scene index — ensuring the same character looks similar across scenes.

### Water Safety System
To prevent SD from generating sea monsters for water-themed stories, the system:
1. Detects water keywords ("ocean", "sea", "lake")
2. Rephrases to safe alternatives ("coastal shoreline background")
3. Includes 30+ creature terms in the negative prompt

### Ken Burns Effect
Applies a slow zoom + pan motion to still images over 2 seconds (50 frames at 25 FPS), creating cinematic video motion.

## Learning Resources

📖 **Full Documentation** — See `/docs/` for comprehensive guides:
- Start with `study_guide.md` for a recommended reading order
- `architecture.md` for system design
- `*_deep_dive.md` files for code walkthroughs
- `ai_models_and_concepts.md` for educational reference

## Troubleshooting

**Sea monsters in ocean scenes?**
→ Increase negative prompt weight or use "coastal shoreline" phrasing

**Generation is slow?**
→ Running on CPU. Install CUDA for GPU acceleration (~10× faster)

**Out of memory?**
→ Enable VAE slicing (already done). On larger systems, use `guidance_scale=9.5` for higher quality.

**Videos not generating?**
→ Check OpenCV codec support. System falls back through mp4v → XVID → MJPG automatically.

## API Reference

### POST `/generate`
Generate a comic strip from a story.
```json
{
  "story": "Luna found the ancient key...",
  "theme": "manga"
}
```
Returns: Array of base64-encoded PNG images

### POST `/generate_video`
Generate an animated video from a story.
```json
{
  "story": "Luna found the ancient key...",
  "style": "cinematic"
}
```
Returns: Path to MP4 file

## Contributing

Contributions welcome! Areas for enhancement:
- Additional art styles
- Multi-character consistency improvements
- Faster model optimization
- Extended language support

## License

MIT License — See LICENSE file for details

## Author

Created by raswanthmalai19

## Acknowledgments

- **Stable Diffusion** by RunwayML
- **NLTK** by NLTK Project
- **Transformers** by Hugging Face
- **PyTorch** by Meta
