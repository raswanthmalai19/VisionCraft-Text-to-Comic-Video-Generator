# VisionCraft — Text to Comic & Video Generator

A Flask web application that transforms stories into comic strips and animated videos using **Stable Diffusion v1.5** and advanced NLP techniques.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda
- ~8GB GPU VRAM (RTX 3050 Ti or better), or CPU (slower)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/text-to-comic.git
   cd text-to-comic
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv svd_env
   source svd_env/bin/activate  # On Windows: svd_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask app:**
   ```bash
   python main.py
   ```

5. **Open in browser:**
   ```
   http://localhost:5000
   ```

---

## 📖 Features

✨ **Comic Generation**
- Converts stories into 4–12 panel comics
- Semantic text chunking for coherent scene grouping
- Theme-based styling (anime, photorealistic, steampunk, etc.)
- PNG output with multi-panel layout

🎬 **Video Generation**
- Creates 2–5 scene cinematic videos
- Ken Burns animation effects (zoom, pan)
- Character consistency across frames
- Water safety system (prevents sea monsters in beach scenes)
- MP4 output with automatic codec fallback

🧠 **AI/ML Pipeline**
- **Stable Diffusion v1.5** — image generation
- **Sentence Transformers** — semantic similarity
- **BART CNN** — text summarization
- **NLTK** — NLP (tokenization, NER, POS tagging)
- **OpenCV** — video assembly

---

## 📁 Project Structure

```
├── main.py                      # Flask server & API routes
├── comic_generator.py           # Core comic generation pipeline (~1400 lines)
├── video_generator.py           # Video generation & Ken Burns effects (~1500 lines)
├── templates/
│   └── index.html              # SPA frontend UI
├── docs/                        # Comprehensive documentation
│   ├── README.md               # Project overview
│   ├── ai_models_and_concepts.md
│   ├── architecture.md
│   ├── comic_generator_deep_dive.md
│   ├── video_generator_deep_dive.md
│   ├── web_interface.md
│   ├── project_structure.md
│   └── study_guide.md
├── output/                      # Generated comics (*.png)
├── video_output/               # Generated videos (*.mp4)
└── svd_env/                    # Virtual environment
```

---

## 🔧 API Endpoints

### POST `/generate` — Generate Comic
**Request:**
```json
{
  "story": "Luna found a mysterious key...",
  "theme": "anime"
}
```

**Response:**
```json
{
  "images": ["data:image/png;base64,...", "..."],
  "filename": "comic_1234567890.png"
}
```

---

### POST `/generate_video` — Generate Video
**Request:**
```json
{
  "story": "Luna found a mysterious key...",
  "style": "cinematic"
}
```

**Response:**
```json
{
  "video_path": "/video_output/video_1234567890.mp4",
  "duration": 12.5
}
```

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[study_guide.md](docs/study_guide.md)** — Recommended reading order + concept cheat sheet
- **[ai_models_and_concepts.md](docs/ai_models_and_concepts.md)** — Deep dives into all ML concepts (15 sections)
- **[architecture.md](docs/architecture.md)** — System architecture & data flow diagrams
- **[comic_generator_deep_dive.md](docs/comic_generator_deep_dive.md)** — Line-by-line breakdown of comic logic
- **[video_generator_deep_dive.md](docs/video_generator_deep_dive.md)** — Line-by-line breakdown of video logic
- **[web_interface.md](docs/web_interface.md)** — Flask API routes & frontend reference
- **[project_structure.md](docs/project_structure.md)** — Every file and folder explained

---

## 🎨 Theme Options

| Theme | Style |
|-------|-------|
| anime | Hand-drawn manga aesthetic |
| photorealistic | Realistic photography |
| steampunk | Victorian gear-powered aesthetic |
| cyberpunk | Neon sci-fi aesthetic |
| watercolor | Soft brush painting |
| oil_painting | Classic fine art |
| comic_book | Bold ink lines with color |
| ukiyo_e | Japanese woodblock print |

---

## 🎬 Video Styles

| Style | Effect |
|-------|--------|
| cinematic | Movie-like with professional lighting |
| animated | Stylized with vibrant colors |
| noir_cinema | Black & white dramatic |
| fantasy_epic | Mythical with rich colors |

---

## ⚙️ Configuration

### Model Loading (lazy by default)
Models are only loaded on the first request to save startup time. To preload:

```python
from comic_generator import load_models
load_models()
```

### GPU vs CPU
Auto-detected. Force CPU:
```python
import torch
# Modify in comic_generator.py/video_generator.py:
device = "cpu"  # instead of torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Guidance Scale (prompt adherence)
```python
# In comic_generator.py
guidance_scale = 7.5   # Default (balanced)
guidance_scale = 9.5   # Stricter (more consistent)
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory (CUDA) | Reduce `num_inference_steps` from 25 to 15 |
| Sea monsters in beach scenes | Already fixed with water word safety system |
| Character looks different in each panel | Increase `guidance_scale` to 9.5 |
| Black/distorted images | Safety checker was disabled (expected on fantasy themes) |
| Video won't load | Check codec fallback (mp4v → XVID → MJPG) |

---

## 📊 Performance Benchmarks

| Task | GPU (RTX 3050 Ti) | CPU |
|------|------------------|-----|
| Comic (6 panels) | ~2 min | ~15 min |
| Video (3 scenes, 2 sec each) | ~4.5 min | ~30+ min |
| Single SD image (512×512) | ~20 sec | ~2.5 min |

---

## 🔬 Technologies

- **Deep Learning:** PyTorch, Diffusers, Transformers
- **NLP:** NLTK, Sentence-Transformers, scikit-learn
- **Computer Vision:** OpenCV (cv2), Pillow (PIL)
- **Web:** Flask, HTML5, CSS3, JavaScript
- **Models:**
  - `runwayml/stable-diffusion-v1-5` (image generation)
  - `all-MiniLM-L6-v2` (semantic similarity)
  - `facebook/bart-large-cnn` (summarization)

---

## 📝 License

MIT License — see LICENSE file for details.

---

## 👤 Author

Created as a learning project in generative AI and NLP.

---

## 💡 Future Improvements

- [ ] Stable Diffusion v2 / v3 support
- [ ] SDXL (1024×1024) high-res output
- [ ] Multi-character dialogue detection
- [ ] Web UI theming options
- [ ] Batch generation API
- [ ] Undo/redo for edited layouts
- [ ] Character model fine-tuning
- [ ] Subtitle generation for videos

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Support

For issues or questions:
- Open a GitHub Issue
- Check `docs/` for detailed documentation
- Review troubleshooting section above

