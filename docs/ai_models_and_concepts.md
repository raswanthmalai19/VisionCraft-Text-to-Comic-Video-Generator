# AI Models & Concepts — Complete Reference

---

## Table of Contents
1. [Stable Diffusion v1.5](#1-stable-diffusion-v15)
2. [Sentence Transformers (all-MiniLM-L6-v2)](#2-sentence-transformers-all-minilm-l6-v2)
3. [BART Large CNN (Summarization)](#3-bart-large-cnn)
4. [NLTK — Natural Language Toolkit](#4-nltk)
5. [TF-IDF Vectorizer](#5-tf-idf-vectorizer)
6. [Cosine Similarity](#6-cosine-similarity)
7. [SemanticChunker — Custom Class](#7-semanticchunker-custom-class)
8. [Encoder-Decoder Architecture](#8-encoder-decoder-architecture)
9. [Prompt Engineering for Stable Diffusion](#9-prompt-engineering-for-stable-diffusion)
10. [Ken Burns Effect (Video Motion)](#10-ken-burns-effect)
11. [Named Entity Recognition (NER)](#11-named-entity-recognition)
12. [POS Tagging](#12-pos-tagging)
13. [Guidance Scale & CFG](#13-guidance-scale--classifier-free-guidance)
14. [Seeding for Reproducibility](#14-seeding-for-reproducibility)
15. [VAE — Variational Autoencoder](#15-vae--variational-autoencoder)

---

## 1. Stable Diffusion v1.5

### What it is
Stable Diffusion (SD) is a **latent diffusion model** for text-to-image generation, developed by RunwayML and trained on LAION-5B. Version 1.5 is the model used here (`runwayml/stable-diffusion-v1-5`).

### How it works (simplified)
1. **Text Encoding:** Your prompt (e.g., "brave knight in a forest, manga style") is tokenized and encoded into a 768-dimensional embedding using a **CLIP text encoder**.
2. **Diffusion:** Starts with pure noise (a random tensor). Over N steps, a **U-Net** gradually denoises the image while being guided by the text embedding.
3. **Decoding:** The U-Net operates in a compressed "latent space." At the end, a **VAE decoder** expands the 64×64 latent into a full 512×512 pixel image.

### How it's used in this project
- **Comic generation:** One 512×512 image per story panel.
- **Video generation:** Three 720×512 images per scene (3 motion variations).

### Key parameters used
```python
pipe(
    prompt,                      # Your engineered text prompt
    negative_prompt,             # What to NOT generate
    num_inference_steps=25,      # How many denoising steps (more = better quality, slower)
    guidance_scale=7.5,          # How strictly to follow the prompt (see section 13)
    width=512, height=512,       # Output resolution
    generator=torch.Generator().manual_seed(seed),  # For reproducibility
    eta=0.0,                     # Stochastic vs deterministic (0.0 = deterministic DDIM)
)
```

### The CLIP Token Limit
CLIP's text encoder has a **77 token limit**. Prompts exceeding this get silently truncated. That's why the video generator limits prompt sections to 50–75 characters each.

---

## 2. Sentence Transformers (all-MiniLM-L6-v2)

### What it is
A lightweight transformer model that converts sentences into **dense semantic vectors** (embeddings) of 384 dimensions. "MiniLM" means it's a distilled (compressed) version of a larger model.

### How it works
- Fine-tuned with **contrastive learning** on sentence pairs.
- Similar sentences produce similar vectors (close in vector space).
- Example:
  - "The knight fought the dragon" → vector₁
  - "The warrior battled the beast" → vector₂
  - cosine_similarity(vector₁, vector₂) ≈ 0.87 (high — same meaning)
  - "She made breakfast" → vector₃
  - cosine_similarity(vector₁, vector₃) ≈ 0.12 (low — different meaning)

### How it's used in this project
```python
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode a list of story sentences
embeddings = sentence_model.encode(["The knight ran.", "He drew his sword.", "She baked bread."])
# Returns: numpy array of shape (3, 384)
```

Then **cosine similarity** between all pairs is computed to group related sentences into the same comic panel.

### Fallback
If `sentence_transformers` is not available, falls back to **TF-IDF** vectors.

---

## 3. BART Large CNN

### What it is
**BART** (Bidirectional and Auto-Regressive Transformers) is an encoder-decoder model by Facebook/Meta. The `facebook/bart-large-cnn` variant is fine-tuned on CNN/DailyMail news summarization.

### How it works
- **Encoder:** Takes the full text, creates contextual representations of every token.
- **Decoder:** Auto-regressively generates a shorter summary, using cross-attention to focus on important parts of the encoder output.

### How it's used in this project
```python
from transformers import pipeline
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")

scene_summary = summarization_model(
    scene_text,
    max_length=60,
    min_length=20,
    do_sample=False   # Greedy decoding — deterministic output
)[0]['summary_text']
```

Used in `generate_scene_description()` to create a concise visual description for each video scene from the raw story text chunk. 

### Fallback
If BART fails to load (common on CPU systems with limited RAM), the system uses **regex + POS-tag based extraction** in `extract_visual_description()`.

---

## 4. NLTK

### What it is
**Natural Language Toolkit** — a comprehensive Python library for NLP, providing pre-built models for:
- **Tokenization** (splitting text into words/sentences)
- **POS tagging** (labeling word types: noun, verb, adjective)
- **NER** (Named Entity Recognition — finding people, places)

### Components used in this project

#### `sent_tokenize`
Splits a paragraph into individual sentences. Uses pre-trained Punkt sentence boundary model.
```python
from nltk.tokenize import sent_tokenize
"Luna found a key. She used it." → ["Luna found a key.", "She used it."]
```

#### `word_tokenize`
Splits a sentence into individual word tokens (handles punctuation separately).
```python
word_tokenize("She ran!") → ["She", "ran", "!"]
```

#### `pos_tag` (Part-of-Speech Tagging)
Labels each word with its grammatical role:
```
[('Luna', 'NNP'), ('found', 'VBD'), ('the', 'DT'), ('key', 'NN')]
NNP = proper noun, VBD = past tense verb, DT = determiner, NN = noun
```

Used to extract:
- **Action verbs** (VB*) → become the "doing word" in prompts
- **Nouns** (NN*) → become visual objects in prompts
- **Adjectives** (JJ*) → paired with nouns for compound phrases like "brass key"

#### `ne_chunk` (Named Entity Recognition)
Identifies Named Entities from POS-tagged text:
```python
from nltk.chunk import ne_chunk
from nltk.tree import Tree

tree = ne_chunk(pos_tags)
# Outputs a tree where named entities are subtrees:
# (PERSON Luna/NNP) → character name
# (GPE London/NNP) → place name
```

Used in `extract_characters_and_setting()` to find character names and locations.

### NLTK Data Downloads
The first time NLTK runs, it downloads required data files:
```python
nltk.download('punkt_tab')       # Sentence tokenizer
nltk.download('stopwords')       # Common words to ignore
nltk.download('averaged_perceptron_tagger')  # POS tagger model
nltk.download('maxent_ne_chunker')           # NER model
nltk.download('words')           # English word list
```

---

## 5. TF-IDF Vectorizer

### What it is
**Term Frequency–Inverse Document Frequency** — a classical NLP technique that converts text into numerical vectors based on word importance.

### How it works

**TF (Term Frequency):** How often does word W appear in document D?
$$TF(w, d) = \frac{\text{count of } w \text{ in } d}{\text{total words in } d}$$

**IDF (Inverse Document Frequency):** How rare is word W across all documents?
$$IDF(w) = \log \frac{\text{total documents}}{\text{documents containing } w}$$

**TF-IDF:**
$$TFIDF(w, d) = TF(w, d) \times IDF(w)$$

Words that appear everywhere (like "the", "a") get low IDF → low score.  
Words that are rare but important (like "sword", "dragon") get high IDF → high score.

### How it's used
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
embeddings = vectorizer.fit_transform(sentences)  # Returns sparse matrix
# Then convert to dense: embeddings.toarray()
```

Acts as a **fallback when sentence_transformers are unavailable**, allowing semantic grouping to still work without the neural model.

---

## 6. Cosine Similarity

### What it is
A measure of similarity between two vectors — specifically, the cosine of the angle between them.

$$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \, |\vec{B}|}$$

Range: **-1 to 1** (for normalized vectors: **0 to 1**)
- 1.0 = identical vectors (same meaning)
- 0.0 = orthogonal (no shared meaning)

### Why cosine (not Euclidean distance)?
Cosine ignores magnitude — so a long paragraph and a short sentence with the same topic get a high similarity score even though their raw vectors have very different lengths.

### How it's used
```python
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)
# similarity_matrix[i][j] = how similar sentence i is to sentence j

# Used in SemanticChunker._group_by_semantics():
if similarity_matrix[i][j] > 0.3:   # Threshold: same semantic group
    group.append(sentence_j)
```

The threshold `0.3` means: if two sentences share at least 30% directional similarity in embedding space, put them in the same comic panel.

---

## 7. SemanticChunker — Custom Class

### What it is
A custom class that orchestrates the full NLP pipeline for breaking a story into **semantically meaningful chunks** (future comic panels).

### Architecture
```
SemanticChunker
│
├── TextEncoder
│     ├─ encode_sentences()     ← SentenceTransformer or TF-IDF
│     └─ _basic_encode_sentences() ← word-count vectors (last resort)
│
├── extract_semantic_units(text)
│     ├─ _analyze_sentences()   ← entities, type, importance score
│     ├─ _group_by_semantics()  ← cosine similarity grouping
│     │     └─ _fallback_grouping() ← by sentence type if cos_sim fails
│     └─ _create_narrative_chunks() ← combine groups, compute metadata
│
└── Chunk output (dict):
      {
        'text': "combined sentence text",
        'importance': 3.5,           # higher = more visually interesting
        'entities': {'people': [], 'places': [], 'things': []},
        'sentence_types': ['action', 'dialogue'],
        'word_count': 12
      }
```

### Sentence Importance Scoring
Each sentence gets a score based on visual richness:
- Length 5–15 words: +2 points
- Visual keywords ("see", "bright", "dark"): +1 each
- Action keywords ("fight", "fly", "attack"): +2 each
- Emotion keywords ("happy", "scared"): +1 each

Higher-importance chunks are less likely to be merged/split.

### Sentence Type Classification
```
"ran / jumped / hit"     → type: 'action'
"He said / She asked"    → type: 'dialogue'  
"was standing / looked"  → type: 'description'
"then / meanwhile"       → type: 'transition'
everything else          → type: 'narrative'
```

---

## 8. Encoder-Decoder Architecture

The codebase frequently references "encoder-decoder architecture." Here's what this means in context:

### In Transformer Models (BART)
```
Input text → [Encoder: bidirectional context] → hidden states
                                                         ↓
                                          [Decoder: autoregressive]
                                                         ↓
                                               Output summary text
```

### In This Project's Custom Pipeline
The project adapts this terminology to describe its own NLP flow:

**"Encoder" role → `TextEncoder` / `SemanticChunker`**
- Takes the raw story text
- Produces high-dimensional semantic representations
- Groups semantically related content

**"Decoder" role → prompt engineering + SD pipeline**
- Takes the semantic representations
- "Decodes" them back into visual descriptions (prompts)
- Then passes those prompts to Stable Diffusion

So when the codebase says "encoder-decoder architecture," it means:
1. **Encode** text → semantic understanding
2. **Decode** semantic understanding → visual image prompts

---

## 9. Prompt Engineering for Stable Diffusion

### What is a Stable Diffusion Prompt?
A text string that tells SD what to generate. SD was trained on image-caption pairs, so it understands natural descriptions, art styles, camera angles, lighting, and quality terms.

### Comic Prompt Structure
```python
# Built by create_enhanced_prompt_v2()
prompt = [
    "Luna clutched the sword",      # Main action (from panel text, enhanced)
    "manga comic panel",             # Theme style
    "luna character",                # Character anchor  
    "forest background",             # Setting
    "bright lighting",               # Visual element (extracted)
    "detailed illustration, high quality"   # Quality boosters
]
# Result: "Luna clutched the sword, manga comic panel, luna character, 
#           forest background, bright lighting, detailed illustration, high quality"
```

### Video Prompt Structure
```python
# Built by create_enhanced_video_prompts()
prompt = [
    "masterpiece, best quality, ultra detailed, sharp focus",   # FIRST — quality anchors
    "adult man, dark brown hair, simple shirt",                  # Character anchor (fixed)
    "clutching, rocky shore, brass key",                         # Scene-specific action+objects
    "coastal shoreline background",                              # Safe setting phrase
    "tense atmosphere",                                          # Mood
    "cinematic film shot, professional movie lighting",          # Style
    "hollywood movie quality",                                   # Quality
    "medium narrative shot",                                     # Camera
    "cinematic framing"                                          # Composition
]
```

### Why "Quality Boosters" First?
Stable Diffusion's CLIP encoder gives more **weight to tokens that appear earlier** in the prompt (due to attention and token position). Putting "masterpiece, best quality" first increases the chance SD pays attention to them.

### Negative Prompts
Tell SD what to **avoid generating**:

```python
# Comic negative prompt
"blurry, low quality, text, watermark, bad anatomy, duplicate"

# Video negative prompt (very comprehensive)
"sea monster, sea creature, kraken, leviathan, mermaid, fish, shark, whale,
 dolphin, octopus, squid, crab, jellyfish, underwater creature, aquatic animal,
 low quality, blurry, distorted, deformed, disfigured, ugly, bad anatomy..."
```

The video negative prompt is so elaborate because Stable Diffusion can confuse "beach", "shore", "ocean" with generating sea creatures. The comprehensive negative list prevents this.

### Water Word Safety Pattern
The video generator has a special pattern to prevent SD from generating sea monsters when the story is set near water:

```python
WATER_WORDS = {'ocean', 'sea', 'lake', 'river', 'water', 'tide', 'shore', ...}

# Instead of: "ocean background"  (might trigger sea creature generation)
# Use:        "coastal shoreline background"  (safer phrasing)

if setting.lower() in WATER_WORDS:
    setting_phrase = "coastal shoreline background"
```

---

## 10. Ken Burns Effect

### What it is
A **video camera technique** that applies slow, smooth zoom and pan movements to still images, creating the illusion of motion. Named after documentary filmmaker Ken Burns.

### Effects implemented
```python
# zoom_in: image appears to grow larger (camera moves forward)
crop shrinks from full size → smaller crop, then resize to target

# zoom_out: opposite (camera pulls back)
crop grows from smaller → full size

# pan_right: camera moves left-to-right across image
x offset increases from 0 → max over duration

# pan_left: opposite
x offset decreases from max → 0
```

### OpenCV implementation
```python
# Scale image to 130% of target resolution
img_scaled = cv2.resize(img, (scaled_w, scaled_h))

for frame_num in range(total_frames):  # total_frames = 25 FPS × 2 seconds = 50 frames
    progress = frame_num / (total_frames - 1)
    
    # For zoom_in:
    current_scale = 1.0 + 0.2 * progress      # grows from 1.0 to 1.2
    crop_w = int(target_w / current_scale)     # crop window shrinks
    cropped = img_scaled[y:y+crop_h, x:x+crop_w]
    frame = cv2.resize(cropped, (target_w, target_h))  # stretch to target
    frames.append(frame)
```

---

## 11. Named Entity Recognition

### What it is
NER identifies and classifies **named entities** in text into categories like:
- PERSON: character names ("Luna", "John")
- GPE (Geopolitical Entity): countries, cities ("London", "France")
- LOCATION: geographical places ("the forest", "Mount Doom")
- ORGANIZATION: companies, groups

### How NLTK NER works
1. **Tokenize** → words
2. **POS Tag** → `[('Luna', 'NNP'), ('forest', 'NN')]`
3. **Chunking** → groups consecutive tagged words by patterns
4. **ne_chunk** → identifies patterns that match known entity types

### How it's used
```python
tree = ne_chunk(pos_tags)
for subtree in tree:
    if isinstance(subtree, Tree):  # It's a named entity!
        entity_name = ' '.join([token for token, pos in subtree.leaves()])
        if subtree.label() == 'PERSON':
            characters.append(entity_name)
        elif subtree.label() in ['GPE', 'LOCATION']:
            settings.append(entity_name)
```

### Character vs. Setting Disambiguation
A major challenge: "Whispering Woods" is a place name, but NLTK might tag "Whispering" as a person name. The project solves this with a large `PLACE_WORDS` set:

```python
PLACE_WORDS = {'woods', 'forest', 'whispering', 'obsidian', 'ancient', ...}

# If the word next to a capitalized word is in PLACE_WORDS,
# treat the whole phrase as a setting, not a character name
if next_word in PLACE_WORDS:
    continue  # skip — it's a place, not a person
```

---

## 12. POS Tagging

### Penn Treebank POS Tags used in this project
| Tag | Meaning | Example |
|-----|---------|---------|
| NNP | Proper noun, singular | Luna, London |
| NNPS | Proper noun, plural | The Avengers |
| NN | Noun, singular | sword, key, forest |
| NNS | Noun, plural | swords, dragons |
| VB | Verb, base form | run, fight, find |
| VBD | Verb, past tense | ran, fought, found |
| VBG | Verb, gerund/present participle | running, fighting |
| VBN | Verb, past participle | defeated, hidden |
| JJ | Adjective | brave, dark, ancient |
| JJR | Adjective, comparative | braver, darker |
| RB | Adverb | quickly, suddenly |
| DT | Determiner | the, a, an |
| IN | Preposition | in, on, at, through |

### How they're used for prompt building
```python
# In extract_visual_description():
for word, tag in pos_tags:
    if tag.startswith('VB') and word not in SKIP_VERBS:
        action_words.append(word)         # "clutched", "ran", "jumped"
    elif tag in ('NN', 'NNS', 'NNP', 'NNPS'):
        key_nouns.append(word)            # "key", "door", "forest"

# In SemanticChunker CompoundPhrase detection:
if tag1.startswith('JJ') and tag2.startswith('NN'):
    compound_phrases.append(f"{word1} {word2}")  # "brass key", "dark forest"
```

---

## 13. Guidance Scale / Classifier-Free Guidance

### What it is
**CFG (Classifier-Free Guidance)** controls how strongly Stable Diffusion follows your text prompt versus generating freely from noise.

$$\text{guided\_direction} = \text{uncond\_output} + \text{guidance\_scale} \times (\text{cond\_output} - \text{uncond\_output})$$

### Effect of different values
| Guidance Scale | Effect |
|---------------|--------|
| 1.0 | Essentially ignores prompt — generates random images |
| 5.0 | Loosely follows prompt — more creative, less accurate |
| 7.5 | **Standard balance** — good prompt adherence + quality |
| 9.5 | Strict prompt following — great for character consistency |
| 15+ | Over-saturated, unnatural — artifacts appear |

### Values used in this project
```python
# Comic panels:
guidance_scale = 7.5   # Standard — good balance

# Video — cinematic style:
guidance_scale = 8.5   # Firmer style adherence

# Video — animated style:
guidance_scale = 9.5   # High — forces consistent character appearance
```

---

## 14. Seeding for Reproducibility

### Why seeds matter
Stable Diffusion starts from **random noise**. The same prompt with different seeds produces completely different images. Using the same seed gives the same image every time.

### Comic panel seeding strategy
```python
story_hash = hash(story[:100]) % 1000       # deterministic per story
panel_seed = (story_hash + i * 137) % 100000  # shifts per panel

generator = torch.Generator(device=device).manual_seed(panel_seed)
```

`137` is a prime number — ensures each panel seed is well-distributed (no clustering).

### Video frame seeding strategy (character consistency)
```python
# Key insight: seed is based on CHARACTER DESCRIPTION, not scene index
story_hash = abs(hash(main_character + char_description)) % 10000
scene_seed = (story_hash + var_idx * 7) % 100000  # scene_idx NOT used!
```

By NOT shifting the seed by `scene_idx`, all scenes start from the same character "space" in SD's internal representation. The same character description with the same seed base → SD generates a more visually consistent person across all scenes.

---

## 15. VAE — Variational Autoencoder

### What it is in Stable Diffusion
SD's pipeline has a VAE (Variational Autoencoder) that compresses images:
- **Encoder:** 512×512 RGB image → 64×64 × 4 latent tensor (factor of 8 compression)
- **Decoder:** 64×64 × 4 latent → 512×512 RGB image

The **U-Net diffusion model works entirely in latent space** (64×64), not pixel space. This is why Stable Diffusion is called a **Latent Diffusion Model** — it's 64× more computationally efficient than pixel-space diffusion.

### VAE slicing (memory optimization)
```python
pipe.enable_vae_slicing()  
# Decodes the VAE in slices instead of all at once
# Reduces peak VRAM usage (helpful for GPUs with < 8GB VRAM)
```

### Why resolution must be divisible by 8
The VAE uses 3 levels of 2× downsampling: 512 → 256 → 128 → 64. If the input is not divisible by 8, the dimensions won't align. That's why:
- Comic panels: 512×512 ✓ (512 = 8 × 64)
- Video frames: 720×512 ✓ (720 = 8 × 90, 512 = 8 × 64)

---

## Summary Table

| Technology | Type | Purpose in Project |
|-----------|------|-------------------|
| Stable Diffusion v1.5 | Latent Diffusion Model | Generate panel/scene images |
| CLIP Text Encoder | Transformer (ViT) | Encode text prompts for SD |
| U-Net | CNN + Attention | Denoise latent space in SD |
| VAE | Convolutional AE | Compress/decompress images |
| all-MiniLM-L6-v2 | Transformer | Sentence semantic embeddings |
| BART-large-CNN | Encoder-Decoder Transformer | Scene description summarization |
| NLTK | Statistical NLP | Tokenization, POS, NER |
| TF-IDF | Classical NLP | Fallback sentence similarity |
| Cosine Similarity | Linear Algebra | Group related sentences |
| SemanticChunker | Custom Class | Orchestrate NLP pipeline |
| OpenCV | Computer Vision | Video frame processing |
| PIL/Pillow | Image Processing | Comic layout, text overlay |
| Flask | Web Framework | HTTP API server |
| PyTorch | Deep Learning | Tensor ops, GPU compute |
