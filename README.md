# 🎨 VisionCraft — Text to Comic & Video Generator

Transform stories into stunning **comic strips** and **animated videos** using AI-powered text-to-image generation and semantic NLP.

## Features

- **📖 Story to Comic:** Convert any story into a multi-panel comic strip with intelligent scene segmentation
- **🎬 Story to Video:** Generate animated videos with consistent character appearance and smooth Ken Burns effects
- **🤖 Advanced NLP:** Semantic chunking, character extraction, entity recognition
- **🎨 Stable Diffusion v1.5:** State-of-the-art diffusion model for high-quality image generation
- **🚀 Web Interface:** Beautiful Flask-based UI with real-time generation
- **💾 Multiple Fallbacks:** Graceful degradation when models or hardware are unavailable

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/VisionCraft.git
   cd VisionCraft
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker')"
   ```

5. **Run the application**
   ```bash
   python main.py
   ```
   
   The Flask server starts at `http://localhost:5000`

### CLI Mode

Generate a comic directly from the terminal:
```bash
python main.py cli
```

## Project Structure

```
VisionCraft/
├── main.py                    # Flask web server & API routes
├── comic_generator.py         # Comic generation pipeline (NLP + SD)
├── video_generator.py         # Video generation & animation
├── templates/
│   └── index.html            # Frontend web interface
├── docs/                     # Comprehensive documentation
│   ├── README.md             # Project overview
│   ├── architecture.md        # System design & data flow
│   ├── ai_models_and_concepts.md  # Deep dive into all models
│   ├── comic_generator_deep_dive.md
│   └── video_generator_deep_dive.md
├── requirements.txt          # Python dependencies
├── .gitignore               # Git exclusions
└── output/                  # Generated comics (gitignored)
```

## How It Works

### Comic Generation
1. **Semantic Analysis** → Break story into meaningful chunks
2. **Character Extraction** → Identify main characters and settings
3. **Prompt Engineering** → Create optimized SD prompts
4. **Image Generation** → Run Stable Diffusion for each panel
5. **Layout Assembly** → Arrange panels into comic pages
6. **Output** → Return as PNG grid

### Video Generation
1. **Scene Extraction** → Divide story into distinct scenes
2. **Character Consistency** → Build character anchor strings
3. **Prompt Crafting** → 9-part structured prompts
4. **Image Synthesis** → Generate 3 motion variations per scene
5. **Animation Effects** → Apply Ken Burns zoom/pan
6. **Video Assembly** → OpenCV video encoding with fallbacks

## API Endpoints

### POST `/generate`
Generate a comic from a story.
```json
{
  "story": "Once upon a time...",
  "theme": "manga"
}
```

### POST `/generate_video`
Generate an animated video.
```json
{
  "story": "Once upon a time...",
  "style": "cinematic"
}
```

### GET `/output/<filename>`
Retrieve a generated comic image.

### GET `/video_output/<filename>`
Retrieve a generated video file.

## Technologies

| Category | Technology |
|----------|-----------|
| Image Generation | Stable Diffusion v1.5 |
| NLP | NLTK, Sentence-Transformers, BART |
| Video | OpenCV, PIL |
| Web | Flask |
| Deep Learning | PyTorch, Transformers |
| Similarity | scikit-learn (TF-IDF, Cosine) |

## Detailed Documentation

For in-depth explanations of every concept, model, and function:

- 📚 [Project Structure](docs/project_structure.md)
- 🏗️ [Architecture & Data Flow](docs/architecture.md)
- 🧠 [AI Models & Concepts](docs/ai_models_and_concepts.md) — 15 deep-dive sections
- 💻 [Comic Generator Breakdown](docs/comic_generator_deep_dive.md)
- 🎬 [Video Generator Breakdown](docs/video_generator_deep_dive.md)
- 🌐 [Web Interface Reference](docs/web_interface.md)
- 📖 [Study Guide](docs/study_guide.md) — Recommended reading order

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU works) | RTX 2060+ or better |
| Storage | 20 GB | 50 GB |

**Note:** GPU acceleration is highly recommended. Generation on CPU takes 5-10 minutes per image; on GPU (RTX 3050 Ti): 20-30 seconds.

## Common Issues & Solutions

### "Sea monsters" appearing near beaches
→ The Water Word Safety system rephrases `"ocean"` to `"coastal shoreline"` to prevent SD from generating creatures. In edge cases, comprehensive negative prompts also help.

### Character looks different in each video frame
→ Use the character anchor system (automatically applied) to maintain consistency. Character seed is based on `hash(character_name + description)`, not scene index.

### Out of memory errors on GPU
→ Models use automatic attention slicing and VAE slicing to reduce VRAM. For GPUs < 4GB, run on CPU or reduce inference steps.

### Generation is very slow
→ On CPU, Stable Diffusion needs 25-35 denoising steps per image. This is normal. Consider using a GPU or reducing `num_inference_steps` to 20.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License — see LICENSE file for details.

## Author

Created as a comprehensive AI-powered storytelling tool combining advanced NLP and generative AI.

---

**For detailed technical documentation, see the [docs/](docs/) folder.**
