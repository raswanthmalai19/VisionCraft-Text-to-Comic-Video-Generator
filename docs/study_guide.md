# Study Guide — How to Learn This Project

---

## Recommended Reading Order

Whether you're a student, developer, or just curious — follow this order for maximum understanding.

---

## Step 1: Understand the Big Picture (20 min)

**Read:** [README.md](README.md)  
**Then read:** [project_structure.md](project_structure.md)

**Key questions to answer after:**
- What does this project actually do?
- What are the 3 main Python files and what does each one do?
- What goes into the `output/` folder? What goes into `video_output/`?
- What is the difference between a comic "panel" and a video "scene"?

---

## Step 2: Understand the Flow (30 min)

**Read:** [architecture.md](architecture.md)

**Focus on:**
1. The **Comic Generation Pipeline** diagram (6 boxes)
2. The **Video Generation Pipeline** diagram (6 boxes)
3. The **lazy model loading** section
4. The **GPU vs CPU** comparison table

**Key questions to answer:**
- What happens step-by-step when a user types a story and clicks "Generate Comic"?
- Why aren't models loaded at server startup?
- What is the difference between running on GPU vs CPU?
- What is the "dependency fallback chain"?

---

## Step 3: Learn the AI Models (45 min)

**Read:** [ai_models_and_concepts.md](ai_models_and_concepts.md)

**Highest priority sections:**
1. **Stable Diffusion v1.5** — the core image generation model
2. **Sentence Transformers** — how text becomes numbers for comparison
3. **Cosine Similarity** — how we measure "are these sentences about the same thing?"
4. **SemanticChunker** — the custom class that uses all of the above
5. **Prompt Engineering** — how we write instructions for Stable Diffusion
6. **Guidance Scale** — the single most important SD parameter to understand

**Key questions to answer:**
- What does Stable Diffusion need as input? What does it produce?
- What is a "latent space"? Why does SD operate in it?
- What does the guidance_scale of 7.5 vs 9.5 mean?
- How do cosine similarity scores of 0.3+ help us group sentences?
- Why does the video generator use seeds based on character description?

---

## Step 4: Dive into the Comic Code (45 min)

**Read:** [comic_generator_deep_dive.md](comic_generator_deep_dive.md)

**Focus on:**
1. The SemanticChunker class (most complex part)
2. `extract_characters_and_setting()` — the NER + disambiguation problem
3. `create_enhanced_prompt_v2()` — prompt engineering in practice
4. `generate_comic()` — the orchestrating function
5. The page layout algorithm

**Key questions to answer:**
- How does the code decide to split a story into 4 panels vs 12 panels?
- Why does `extract_characters_and_setting` have a `PLACE_WORDS` set?
- What does `enhance_panel_description_v2` do to the text "was happy"?
- How is the comic PNG assembled from individual panel images?

---

## Step 5: Dive into the Video Code (45 min)

**Read:** [video_generator_deep_dive.md](video_generator_deep_dive.md)

**Focus on:**
1. `build_character_description()` — the character anchor system
2. `create_enhanced_video_prompts()` — 9-part structured prompt
3. The Water Word Safety system
4. `apply_animation_effects()` — the Ken Burns effect math
5. `create_video_with_transitions()` — OpenCV crossfade assembly

**Key questions to answer:**
- Why does the video use `"coastal shoreline background"` instead of `"ocean background"`?
- How does the Ken Burns zoom-in work mathematically?
- Why is the scene seed based on character description instead of scene index?
- What is the crossfade algorithm and which OpenCV function implements it?

---

## Step 6: Understand the Web Layer (20 min)

**Read:** [web_interface.md](web_interface.md)

**Focus on:**
- The 4 Flask routes and what JSON they accept/return
- How the frontend renders base64 images without touching the file system
- The timeout configurations and why they differ

---

## Concept Cheat Sheet

| Concept | One-Line Explanation |
|---------|---------------------|
| Stable Diffusion | AI that converts a text prompt into a 512×512 image by denoising random noise |
| CLIP | The sub-model inside SD that converts text prompts to embeddings |
| Latent Space | A compressed (64×64) mathematical representation of an image |
| U-Net | The neural network that does the actual denoising in SD |
| VAE | Compresses images 8× for SD's U-Net, then expands them back |
| Guidance Scale | How strictly SD follows your prompt (7.5 = standard, 9.5 = strict) |
| Negative Prompt | Words you tell SD to avoid in the output |
| Semantic Embedding | A vector of numbers that captures the "meaning" of a sentence |
| Cosine Similarity | Score 0–1 measuring how similar two embedding vectors are |
| TF-IDF | Classical (non-neural) way to represent documents as word-importance vectors |
| Named Entity Recognition | Automatically finding people and place names in text |
| POS Tagging | Labeling each word as noun/verb/adjective/etc. |
| SemanticChunker | The custom class that groups similar sentences into comic panels |
| Character Anchor | A fixed description of the character injected into every SD prompt for consistency |
| Ken Burns Effect | Applying slow zoom/pan movement to still images to create video motion |
| Crossfade | Blending frames from two consecutive scenes so the transition is smooth |
| Lazy Loading | Not loading models until the first request (saves startup time) |
| Encoder-Decoder | Architecture where text is first "encoded" to meaning, then "decoded" to images |

---

## Common Questions

**Q: Why does generation take so long?**  
A: Stable Diffusion runs 25–35 denoising steps per image. Each step involves a full forward pass through a ~860M parameter U-Net. For a 6-panel comic: 6 images × 25 steps = 150 U-Net forward passes. On CPU this is very slow (~10 min/image). On GPU (RTX 3050 Ti): ~20–30 sec/image.

**Q: Why do the same story inputs sometimes produce different images?**  
A: Seeds are based on a hash of the first 100 characters of the story + the panel index. If the story text is the same, you'll get the same images. If you change even one word, the hash changes, and a different seed is used.

**Q: Why does the video sometimes show "sea monsters" even when the story is about a person near the ocean?**  
A: Stable Diffusion v1.5 was trained on internet images, so prompts with words like "ocean", "sea", "tide" can trigger underwater/creature imagery. The codebase addresses this with the Water Word Safety system but edge cases can slip through.

**Q: What happens if I have no GPU?**  
A: The code detects `torch.cuda.is_available()` and switches to CPU mode automatically, using float32 instead of float16. Generation still works but takes much longer.

**Q: Why is the video resolution 720×512 and not 1920×1080?**  
A: Stable Diffusion v1.5 was trained on 512×512 images. Going much larger introduces artifacts. 720×512 is a compromise between wider video format and staying close to SD's native resolution. Also, higher resolution = more VRAM needed = out of memory errors on typical GPUs.

---

## Key Design Patterns to Study

### 1. Graceful Degradation
Every major feature has a fallback:
```
TRANSFORMERS → TFIDF → basic word vectors
NLTK → regex splitting
BART → NLP regex extraction
GPU → CPU
```

### 2. Lazy Loading
```python
if pipe is None:
    pipe = StableDiffusionPipeline.from_pretrained(...)
```
Learn why this pattern exists (cold start optimization).

### 3. Deterministic Seeding
```python
seed = (hash(story[:100]) + panel_idx * 137) % 100000
```
Using prime number multipliers (`137`, `7`) to avoid seed clustering.

### 4. Prompt Engineering as a Science
Read how prompts are built in 9 ordered parts, why quality boosters go first, and how negative prompts work.

### 5. Character Consistency Across Frames
The same character description string is injected into every single SD prompt. This is a heuristic approach — it doesn't guarantee identical appearance but significantly improves consistency compared to not doing it.
