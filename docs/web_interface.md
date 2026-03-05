# Web Interface & API Reference

---

## Overview

The web interface is a **single-page application (SPA)** built with:
- **Backend:** Flask (Python) — `main.py`
- **Frontend:** Vanilla HTML + CSS + JavaScript — `templates/index.html`

No JavaScript framework (React/Vue/Angular) is used — everything is plain browser-native code.

---

## Flask Routes (Backend API)

### `GET /`
**Template:** `templates/index.html`  
**Purpose:** Serves the main web application. Flask's `render_template()` loads the Jinja2 template.

---

### `POST /generate` — Comic Generation
**Request Body (JSON):**
```json
{
  "story": "A brave knight found a magical sword...",
  "theme": "classic"
}
```

**Validation:**
- `story` must be non-empty string
- `theme` must be one of the 8 known themes (falls back to `'classic'` if unknown)

**Processing:**  
Calls `generate_comic(story, theme)` which returns:
```python
comic_paths    # list of PNG file paths on disk
imgs_base64    # list of base64-encoded PNG strings
num_panels     # total panel count
layout_info    # dict with pages breakdown
```

**Response (JSON — Success):**
```json
{
  "success": true,
  "images": ["iVBORw0KGgo...", "..."],   // base64 PNGs (one per page)
  "paths": ["output/semantic_comic_page_1.png", ...],
  "theme": "Classic Comic Book",
  "panels": 6,
  "pages": 1,
  "story_analysis": {
    "total_panels": 6,
    "total_pages": 1,
    "page_layouts": [[6, "3x2"]]
  }
}
```

**Response (JSON — Error):**
```json
{
  "error": "Please enter a story"
}
```
Status: 400 (bad input) or 500 (generation error)

---

### `POST /generate_video` — Video Generation
**Request Body (JSON):**
```json
{
  "story": "A brave knight found a magical sword...",
  "style": "cinematic"
}
```

**Validation:**
- `story` must be non-empty
- `story` must have at least 20 words (minimum for scene extraction)

**Processing:**  
Calls `generate_video_from_story(story, style)` which returns a dict containing `video_path`.

**Response (JSON — Success):**
```json
{
  "success": true,
  "video_path": "/video_output/story_video_20260305_141523.mp4",
  "video_filename": "story_video_20260305_141523.mp4",
  "num_scenes": 7,
  "story_analysis": {
    "scenes": 7,
    "summary": "Director's Cut video with 7 cinematic scenes"
  },
  "duration_estimate": "14 seconds",
  "style": "cinematic"
}
```

---

### `GET /themes` — List Available Themes
**Response:**
```json
{
  "themes": {
    "classic": "Classic Comic Book",
    "manga": "Manga/Anime Style",
    "superhero": "Superhero Comic",
    "cartoon": "Cartoon Style",
    "noir": "Film Noir",
    "fantasy": "Fantasy Adventure",
    "scifi": "Science Fiction",
    "webcomic": "Modern Web Comic"
  }
}
```

---

### `GET /output/<filename>` — Serve Comic Files
Serves static PNG files from the `output/` directory.  
Used internally by Flask but not directly called by the frontend (comics are sent inline as base64).

---

### `GET /video_output/<filename>` — Serve Video Files
The most complex route — validates the video before serving.

```python
@app.route('/video_output/<filename>')
def video_output_file(filename):
    video_path = os.path.join(VIDEO_OUTPUT_DIR, filename)
    
    # Validate with OpenCV
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if frame_count == 0:
        return "Video file is corrupted or has no content", 500
    
    # Set streaming-friendly headers
    response = send_from_directory(VIDEO_OUTPUT_DIR, filename)
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'       # Enables seek/scrub
    response.headers['Cache-Control'] = 'public, max-age=3600'
    response.headers['Content-Disposition'] = 'inline; filename="' + filename + '"'
    return response
```

**Why validate the video?**  
OpenCV's VideoWriter can create empty or corrupted files if the codec fails. This check prevents the browser from trying to play an empty file (which shows a broken video player).

---

## Frontend Structure (index.html)

### HTML Layout
```html
<div class="container">
  <!-- HEADER: "VisionCraft" title -->
  <div class="header"> ... </div>
  
  <div class="main-content">
    <!-- STORY INPUT -->
    <textarea id="storyInput"> ... </textarea>
    
    <!-- THEME SELECTOR (8 options) -->
    <select id="themeSelector"> ... </select>
    <div id="themePreview"> ... </div>
    
    <!-- COMIC GENERATE BUTTON -->
    <button id="generateBtn" onclick="generateComic()"> ... </button>
    
    <!-- VIDEO SECTION -->
    <div class="video-section">
      <select id="videoStyleSelector"> ... </select>
      <button id="generateVideoBtn" onclick="generateVideo()"> ... </button>
    </div>
    
    <!-- LOADING STATE -->
    <div id="loading"> ... spinner + text ... </div>
    
    <!-- RESULTS (dynamically filled) -->
    <div id="result"> ... </div>
    
    <!-- EXAMPLE STORIES -->
    <div class="example-stories"> 3 clickable examples </div>
  </div>
</div>
```

---

### CSS Highlights

**Color scheme:**
- Background gradient: `#667eea → #764ba2` (purple)
- Header gradient: `#FF6B6B → #4ECDC4` (coral → teal)
- Video section: same purple gradient as body background

**Loading spinner:**
```css
.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    width: 50px; height: 50px;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    0%   { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
```

---

### JavaScript Functions

#### `generateComic()`
```javascript
async function generateComic() {
    // 1. Validate story input
    // 2. Disable button, show loading spinner
    // 3. Set 10-minute AbortController timeout
    // 4. POST /generate with {story, theme}
    // 5. On success: render comic pages as <img> tags with base64 src
    //    - If multi-page: show "Page N of M" headers + download buttons
    //    - If single-page: show single image + download button
    // 6. On error: show red error div
    // 7. Finally: re-enable button, hide spinner
}
```

**Rendering multi-page comics:**
```javascript
data.images.forEach((image, index) => {
    pageHtml += `
        <div class="page-container">
            <h4>Page ${index + 1} of ${data.pages}</h4>
            <img src="data:image/png;base64,${image}" class="comic-image" />
            <button onclick="downloadImage('${image}', 'comic_page_${index+1}.png')">
                📥 Download Page ${index + 1}
            </button>
        </div>`;
});
```

#### `generateVideo()`
```javascript
async function generateVideo() {
    // 1. Validate: story must have >= 20 words
    // 2. Set 20-minute AbortController timeout
    // 3. Show rotating progress messages every 2 minutes:
    //    "AI is analyzing your story structure..."
    //    "Generating high-quality images with Stable Diffusion..."
    //    etc.
    // 4. POST /generate_video with {story, style}
    // 5. On success: render <video> element with src={video_path}
    //    - Also set URL HEAD test to debug accessibility
    // 6. Debug event listeners on <video>:
    //    onloadstart, oncanplay, onerror, onloadedmetadata, onplay
}
```

**Video element (rendered dynamically):**
```html
<video controls preload="metadata" style="width:100%; max-width:600px"
       onerror="console.error('Video error:', this.error)">
  <source src="/video_output/story_video_20260305_141523.mp4" type="video/mp4">
  <source src="..." type="video/webm">
  <source src="..." type="video/ogg">
  <p>Your browser does not support video. <a href="...">Download</a></p>
</video>
```

Multiple `<source>` fallbacks ensure the video plays in as many browsers as possible.

#### `downloadImage(base64Data, filename)`
```javascript
function downloadImage(base64Data, filename) {
    const link = document.createElement('a');
    link.href = 'data:image/png;base64,' + base64Data;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
```

Creates a hidden `<a>` element with a data URL, clicks it programmatically, then removes it. This triggers a browser file download without a page navigation.

#### `downloadAllPages(imagesArray)`
```javascript
imagesArray.forEach((image, index) => {
    setTimeout(() => {
        downloadImage(image, `comic_page_${index + 1}.png`);
    }, index * 200);   // 200ms stagger to avoid browser blocking
});
```

Staggered downloads prevent browsers from blocking multiple simultaneous download triggers.

#### `useExample(element)`
```javascript
function useExample(element) {
    document.getElementById('storyInput').value = element.textContent.trim();
}
```

Clicking an example story div copies its text into the textarea.

---

## Error Handling Strategy

### Backend (Flask)
```python
try:
    result = generate_comic(story, theme)
    return jsonify({'success': True, ...})
except Exception as e:
    print(f"Error: {str(e)}")                # Logged to console
    return jsonify({'error': str(e)}), 500   # Shown to user
```

### Frontend (JavaScript)
```javascript
try {
    const response = await fetch('/generate', { ..., signal: controller.signal });
    const data = await response.json();
    if (data.success) { /* render */ }
    else { result.innerHTML = `<div class="error">❌ ${data.error}</div>`; }
} catch (error) {
    if (error.name === 'AbortError') {
        // Timeout message with suggestions
    } else {
        // Network error message
    }
} finally {
    // Always: re-enable buttons, hide spinner
}
```

---

## Timeout Configuration

| Generation Type | Timeout | Reasoning |
|----------------|---------|-----------|
| Comic | 10 minutes (600,000 ms) | Long stories can take 5-10 min on GPU |
| Video | 20 minutes (1,200,000 ms) | 3 variations × multiple scenes × 35 steps |

---

## Content Security Notes

The application:
- Does NOT use parameterized inputs for any shell commands (no injection risk)
- Does NOT execute any user-provided code
- Serves files only from the `output/` and `video_output/` directories (no path traversal — Flask's `send_from_directory` handles this)
- The AI-generated content flows through Python → base64 → JSON → browser data URL (no file system path is exposed to the browser for comic images)
