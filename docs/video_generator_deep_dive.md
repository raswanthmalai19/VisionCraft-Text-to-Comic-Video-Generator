# video_generator.py — Complete Deep Dive

---

## File Purpose
`video_generator.py` handles the **video generation** pipeline. It reuses the comic generator's NLP and image generation infrastructure, but extends it with:
- **Scene-based** (not panel-based) story segmentation
- **Character anchor** system for visual consistency across frames
- **Motion effects** (Ken Burns — zoom/pan)
- **Video assembly** with crossfade transitions via OpenCV

---

## Imports & Dependencies

```python
import torch          # PyTorch
import cv2            # OpenCV — video frame manipulation
import numpy as np    # Array operations
import json           # JSON parsing (legacy)
import os, math
from datetime import datetime   # For timestamped filenames
```

### Conditional imports
```python
from diffusers import StableDiffusionPipeline   # (same as comic_generator)
from moviepy import ImageSequenceClip, ...       # Optional — higher-level video
```

### Critical import from comic_generator
```python
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
    DIFFUSERS_AVAILABLE
)
import comic_generator  # Also imported as module for accessing live globals (pipe, etc.)
```

This tight coupling means `video_generator.py` launches the same model loading as the comic generator — models are shared between both pipelines.

---

## Configuration Constants

```python
VIDEO_OUTPUT_DIR = "video_output"
FRAME_RATE = 25           # Frames per second in the output MP4
IMAGE_DURATION = 2        # Each image's screen time in seconds (= 50 frames)
TRANSITION_DURATION = 0.5 # Crossfade duration at scene boundaries (= 12 frames)
VIDEO_RESOLUTION = (720, 512)  # Width × Height (must be divisible by 8 for SD)
```

### Video Styles (4 options)
```python
VIDEO_STYLES = {
    'cinematic': {
        'visual':  'cinematic film shot, professional movie lighting, film grain',
        'mood':    'dramatic cinematic atmosphere',
        'camera':  'professional film camera work, shallow depth of field',
        'quality': 'hollywood movie quality, photorealistic'
    },
    'animated': {
        'visual':  'high quality 3D animation, pixar movie style, studio ghibli quality',
        'mood':    'emotional animated atmosphere, cinematic lighting',
        'camera':  'cinematic animated camera work',
        'quality': 'pixar feature film quality, studio ghibli animation'
    },
    'realistic': { ... },   # documentary photography style
    'artistic':  { ... }    # fine art painting style
}
```

---

## Scene Extraction Pipeline

### `extract_story_timeline(story_text, style)` — Entry Point
Decides how many scenes to create based on word count:
```python
if word_count < 30:    target_scenes = 3
elif word_count < 80:  target_scenes = 5
elif word_count < 150: target_scenes = 7
elif word_count < 300: target_scenes = 10
else:                  target_scenes = min(15, ceil(word_count / 30))
```

Then delegates to `extract_scenes_with_transformers()` (primary) or `create_timeline_fallback()`.

---

### `extract_scenes_with_transformers(story_text, target_scenes, style)`
**6-step transformer-based pipeline:**

**Step 1: Global Context**
```python
chars, settings = extract_characters_and_setting(story_text)
main_char = chars[0]     # "Luna"
main_setting = settings[0]  # "forest"
char_description = build_character_description(story_text, main_char)
# → "adult woman, dark brown hair, simple shirt and trousers"
```

**Step 2: Semantic Chunking**
```python
chunker = SemanticChunker()
semantic_chunks = chunker.extract_semantic_units(story_text)
# → list of dicts with 'text', 'importance', 'entities', etc.
```

**Step 3: Distribute Chunks into target_scenes**
```python
chunks_per_scene = max(1, len(semantic_chunks) // target_scenes)
for scene_idx in range(target_scenes):
    scene_chunks = semantic_chunks[start_idx:end_idx]
    scene_text = " ".join([c['text'] for c in scene_chunks])
```

**Step 4: Generate Scene Visual Description**
```python
visual_desc = generate_scene_description(scene_text, main_char, main_setting, style)
```

**Step 5: Extract Mood**
```python
mood = extract_mood(scene_text)
# → 'joyful' | 'melancholic' | 'tense' | 'mysterious' | 'peaceful' | 'dramatic' | ...
```

**Step 6: Assemble Scene Dict**
```python
scenes.append({
    "scene_number": scene_idx + 1,
    "scene_title": generate_scene_title(scene_text, scene_idx+1, main_char),
    "visual_description": visual_desc,
    "mood": mood,
    "key_elements": extract_key_elements(scene_text, main_char)
})
```

**Returns:**
```python
{
    "story_metadata": {
        "main_character": "Luna",
        "char_description": "adult woman, dark brown hair, simple shirt",
        "setting": "forest",
        "visual_style": "cinematic"
    },
    "scenes": [
        {"scene_number": 1, "scene_title": "...", "visual_description": "...", "mood": "tense", "key_elements": [...]},
        ...
    ]
}
```

---

## Character Anchor System: `build_character_description()`

A critical innovation for video consistency. By injecting the **same character description** into every SD prompt, the model is encouraged to generate the same-looking person in every frame.

### Detection logic:
```python
# Gender detection (pronoun counting)
if count('he ') + count('his ') + count('him ') >= count('she ') + count('her '):
    gender = 'man'
else:
    gender = 'woman'

# Age detection (keyword presence)
if any(['boy ', 'girl ', 'teen', 'young'] in text):   age = 'young'
elif any(['old ', 'elderly', 'aged '] in text):        age = 'elderly'
else:                                                   age = 'adult'     # default

# Hair detection (keyword mapping)
hair_map = [
    ('dark hair',  ['dark hair', 'black hair', 'dark-haired']),
    ('blond hair', ['blond', 'blonde', 'golden hair']),
    ('red hair',   ['red hair', 'auburn', 'ginger']),
    ('grey hair',  ['grey hair', 'silver hair', 'white hair']),
    ('brown hair', []),   # This is the default
]

# Clothing detection
clothing_map = [
    ('heavy coat',  ['coat', 'overcoat', 'trench coat']),
    ('dark cloak',  ['cloak', 'cape']),
    ('armor',       ['armor', 'armour']),
    ...
    ('simple shirt and trousers', []),  # default
]

# Story-specific props
if re.search(r'\b(key|brass key|small key)\b', text_lower):
    props.append('holding a small brass key')

# Final assembly
return "adult man, dark brown hair, simple shirt and trousers, holding a small brass key"
```

This string is stored in `story_metadata['char_description']` and injected into **every single scene prompt**.

---

## Scene Description Generation

### `generate_scene_description(scene_text, main_char, main_setting, style)`
```python
# Primary: use BART summarization if loaded
if summarization_model is not None:
    summary = summarization_model(
        scene_text,
        max_length=60, min_length=20,
        do_sample=False       # greedy/deterministic
    )[0]['summary_text']
    visual_desc = f"{summary}, {style_desc['visual'][:30]}, featuring {main_char}"
    return visual_desc

# Fallback: NLP-based extraction
return extract_visual_description(scene_text, main_char, main_setting, style_desc)
```

### `extract_visual_description()` — NLP Fallback (the real workhorse)
When BART is unavailable, this function extracts scene-specific visual descriptions using pure NLP:

**Step 1: Camera angle selection**
```python
camera_angles = ["wide shot", "medium shot", "close-up shot", "establishing shot"]
camera = camera_angles[hash(scene_text[:30]) % 4]
```
Uses a hash of the scene's first 30 characters for deterministic but varied camera selection.

**Step 2: NLTK POS extraction**
```python
for sent in sentences[:4]:
    tokens = word_tokenize(sent)
    pos_tags = pos_tag(tokens)
    for word, tag in pos_tags:
        if tag.startswith('VB') and word not in SKIP_VERBS:
            action_words.append(word)      # e.g., "clutched", "trekked"
        elif tag in ('NN', 'NNS', 'NNP', 'NNPS'):
            key_nouns.append(word)         # e.g., "key", "cliff", "mist"
```

**Step 3: Compound phrase detection (adjective+noun)**
```python
for j in range(len(all_tokens) - 1):
    w1, t1 = all_tokens[j]
    w2, t2 = all_tokens[j+1]
    if t1.startswith('JJ') and t2.startswith('NN'):
        compound_phrases.append(f"{w1} {w2}")   # "brass key", "salt spray"
```

**Step 4: Story-specific regex matching**
```python
story_phrase_hits = re.findall(
    r'\b(brass key|small key|barnacle-encrusted rock|frozen waves|'
    r'salt spray|grandfather.s voice|faint glimmer|...)\b',
    scene_text.lower()
)
```

**Step 5: Location extraction**
```python
loc_matches = re.findall(
    r'(?:at|in|on|beneath|under|through|across|into|upon|over|near|along)\s+'
    r'(?:the\s+)?([\w]+(?:\s+[\w]+){0,3})',
    scene_text.lower()
)
```
Looks for prepositional phrases like "at the cliff", "beneath the tide" to get the scene-specific location.

**Step 6: Assembly**
```python
desc_parts = [
    f"wide shot of {main_char} clutching",    # camera + character + action
    "rocky shore",                             # location
    "brass key, frozen waves, cliff",          # compound phrases + nouns
    "cinematic film shot"                      # style (clipped to 45 chars)
]
return ', '.join(desc_parts)
# → "wide shot of Luna clutching, rocky shore, brass key, frozen waves, cinematic film shot"
```

---

## Mood Extraction: `extract_mood()`

Keyword scoring across 8 mood categories:
```python
mood_mappings = {
    'joyful':      ['happy', 'joy', 'laugh', 'smile', 'bright', 'cheerful'],
    'melancholic': ['sad', 'sorrow', 'tear', 'cry', 'lonely', 'grief'],
    'tense':       ['danger', 'threat', 'fear', 'afraid', 'nervous', 'anxious'],
    'mysterious':  ['strange', 'odd', 'mysterious', 'curious', 'secret', 'hidden'],
    'peaceful':    ['calm', 'quiet', 'serene', 'peaceful', 'gentle', 'soft'],
    'dramatic':    ['sudden', 'shock', 'gasp', 'realize', 'discover', 'reveal'],
    'romantic':    ['love', 'heart', 'tender', 'embrace', 'kiss', 'passion'],
    'adventurous': ['journey', 'quest', 'explore', 'discover', 'venture', 'brave']
}
# Returns: highest-scoring mood, default 'dramatic'
```

---

## Video Prompt Engineering: `create_enhanced_video_prompts()`

The most sophisticated function in the video pipeline. Builds a 9-part prompt:

```
1. "masterpiece, best quality, ultra detailed, sharp focus"
   └─ Quality anchors — placed FIRST for SD's positional attention bias

2. char_anchor (fixed per story)
   └─ e.g. "adult man, dark brown hair, heavy coat, holding a brass key"

3. scene_specific
   └─ Action + objects extracted from visual_description (camera prefix stripped)

4. setting_phrase
   └─ Safe phrasing (water settings become "coastal shoreline background")

5. f"{mood} atmosphere"
   └─ e.g. "tense atmosphere", "mysterious atmosphere"

6. style_keywords (first 50 chars of style visual)
   └─ e.g. "cinematic film shot, professional movie lighting"

7. quality (first 40 chars of style quality)
   └─ e.g. "hollywood movie quality"

8. camera_note (progressive across scenes)
   └─ first 15% of story: "establishing wide shot"
      middle 70%:          "medium narrative shot"
      last 15%:            "dramatic close-up"

9. "cinematic framing"
```

### Water Word Safety System
To prevent SD from generating sea monsters near water settings:
```python
WATER_WORDS = {'ocean', 'sea', 'lake', 'river', 'water', 'tide', 'shore', 'coast', ...}
CREATURE_TRIGGER_WORDS = WATER_WORDS | {'barnacle', 'barnacles', 'seaweed', 'coral', ...}

# Strip water words from scene_specific (prevents: "...waves, salt spray")
scene_specific_parts = [
    token for token in scene_specific.split(', ')
    if token.strip().lower() not in CREATURE_TRIGGER_WORDS
]

# Rephrase water settings
if setting.lower() in WATER_WORDS:
    setting_phrase = "coastal shoreline background"  # safe
else:
    setting_phrase = f"{setting} environment background"
```

### Comprehensive Negative Prompt
```python
creature_negative = (
    "sea monster, sea creature, ocean monster, kraken, leviathan, sea serpent, "
    "mermaid, fish, shark, whale, dolphin, octopus, squid, crab, jellyfish, "
    "underwater creature, aquatic animal, fantasy creature, monster, beast, "
    "animal, wildlife, alien creature, dragon, demon, tentacles, scales, fins"
)

style_negatives = {
    'cinematic': "cartoon, animated, anime, 3D render, cel-shaded, disney, pixar",
    'animated':  "photorealistic, live action, realistic photography, film grain, real person",
    'realistic': "cartoon, animated, anime, 3D render, cel-shaded, illustration, drawing",
    'artistic':  "photorealistic, live action, realistic photography, film grain, real person"
}
# Prevents style contamination (e.g., cinematic style getting cartoon elements)
```

---

## Image Generation (Per Scene × 3 Variations)

```python
for scene_idx, scene in enumerate(scenes):
    for var_idx in range(3):
        # Character-stable seeding (NOT scene_idx-dependent)
        story_hash = abs(hash(main_character + char_description)) % 10000
        scene_seed = (story_hash + var_idx * 7) % 100000
        
        # Style-specific generation parameters
        if style == 'animated':
            guidance_scale = 9.5   # high — forces character consistency
            num_steps = 35
        elif style == 'cinematic':
            guidance_scale = 8.5
            num_steps = 35
        else:
            guidance_scale = 8.0
            num_steps = 30
        
        image = image_pipe(
            image_prompt,
            negative_prompt=enhanced_negative,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            width=720, height=512,
            generator=generator,
            eta=0.0
        ).images[0]
        
        # Apply motion effect based on variation index
        motion = ["static", "zoom_in", "zoom_out"][var_idx]
        frames = apply_animation_effects(temp_path, motion, IMAGE_DURATION)
        all_frames.extend(frames)
    
    # Clear GPU VRAM every 3 scenes
    if (scene_idx + 1) % 3 == 0:
        torch.cuda.empty_cache()
```

---

## Motion Effects: `apply_animation_effects()`

### Algorithm
```python
# Scale image up by 1.3× to have room to move
img_scaled = cv2.resize(img, (scaled_w, scaled_h))
# scaled_w = 720 × 1.3 = 936
# scaled_h = 512 × 1.3 = 665.6 → 666

total_frames = IMAGE_DURATION × FRAME_RATE = 2 × 25 = 50 frames per image
```

### zoom_in (frames 0→49, scale 1.0→1.2)
```python
for frame_num in range(50):
    progress = frame_num / 49              # 0.0 → 1.0
    current_scale = 1.0 + 0.2 * progress  # 1.0 → 1.2
    
    # Shrinking crop window (center-anchored)
    crop_w = int(720 / current_scale)      # 720 → 600
    crop_h = int(512 / current_scale)      # 512 → 427
    start_x = (936 - crop_w) // 2
    start_y = (666 - crop_h) // 2
    
    cropped = img_scaled[start_y:start_y+crop_h, start_x:start_x+crop_w]
    frame = cv2.resize(cropped, (720, 512))  # stretch to target
    frames.append(frame)
```

The crop window shrinks → the same content occupies more pixels → **appears to zoom in**.

---

## Video Assembly: `create_video_with_transitions()`

### Codec selection (tries in order)
```python
codecs_to_try = [
    cv2.VideoWriter_fourcc(*'mp4v'),   # MPEG-4 Part 2 — most compatible
    cv2.VideoWriter_fourcc(*'XVID'),   # Xvid
    cv2.VideoWriter_fourcc(*'MJPG'),   # Motion JPEG — always works, larger files
]
for fourcc in codecs_to_try:
    out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, VIDEO_RESOLUTION)
    if out.isOpened():
        break  # Use first successful codec
```

### Crossfade transitions
```python
frames_per_scene = len(all_frames) // num_scenes
transition_frames = int(0.5 × 25) = 12  # 12 frames of crossfade

for i, frame in enumerate(all_frames):
    if i % frames_per_scene < transition_frames:
        # Blend current frame with previous frame
        alpha = (i % frames_per_scene) / transition_frames   # 0.0 → 1.0
        blended = cv2.addWeighted(prev_frame, 1-alpha, frame, alpha, 0)
        out.write(blended)
    else:
        out.write(frame)
```

### Output path
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = f"video_output/story_video_{timestamp}.mp4"
```

### Validation
Post-write check to ensure the file is real:
```python
file_size = os.path.getsize(video_path)
if file_size < 1000:   # bytes
    raise RuntimeError("Video file too small — encoding error")
```

---

## Main Entry: `generate_video_from_story(story, style)`

```python
def generate_video_from_story(story, style='cinematic'):
    # 1. Extract scenes using transformer pipeline
    director_analysis = extract_story_timeline(story, style)
    
    # 2. Validate structure (type safety)
    if not isinstance(director_analysis, dict):
        director_analysis = create_timeline_fallback(story, 5, style)
    
    story_metadata = director_analysis['story_metadata']
    scenes = director_analysis['scenes']
    
    # 3. Load Stable Diffusion (gets the live pipe object, not the None import)
    image_pipe = load_models()    # Returns the loaded pipe
    image_pipe.safety_checker = None   # Disable to prevent black frames
    
    # 4. Generate 3 images per scene with applied motion
    all_frames = []
    for scene_idx, scene in enumerate(scenes):
        for var_idx in range(3):
            # ... generate image, apply motion, extend all_frames
    
    # 5. Assemble video
    video_path = create_video_with_transitions(all_frames, len(scenes))
    
    # 6. Clean up temp PNGs
    for temp_img in temp_images:
        os.remove(temp_img)
    
    return {
        'success': True,
        'video_path': video_path,
        'total_scenes': len(scenes),
        'total_frames': len(all_frames),
        'story_analysis': { 'scenes': scenes, 'summary': "..." },
        'style': style
    }
```

---

## Performance Summary: Comic vs Video

| Metric | Comic | Video |
|--------|-------|-------|
| Image resolution | 512×512 | 720×512 |
| Images generated | 1 per panel | 3 per scene |
| Inference steps (GPU) | 25 | 35 |
| Guidance scale | 7.5 | 8.5–9.5 |
| Output format | PNG pages | MP4 file |
| Motion effects | None | zoom_in / zoom_out / static |
| Transitions | None | crossfade |
| VRAM needed | ~4 GB | ~5 GB |
| GPU time (10-panel story) | ~5 min | ~15–20 min |
