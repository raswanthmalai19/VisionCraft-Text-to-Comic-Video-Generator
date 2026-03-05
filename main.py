from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import traceback

# Import our custom modules with error handling
try:
    from comic_generator import generate_comic, COMIC_THEMES, OUTPUT_DIR, DIFFUSERS_AVAILABLE
    COMIC_AVAILABLE = True
except Exception as e:
    print(f"❌ Error importing comic_generator: {e}")
    print("⚠️ Comic generation will be disabled")
    COMIC_AVAILABLE = False
    OUTPUT_DIR = "output"
    COMIC_THEMES = {}

try:
    from video_generator import generate_video_from_story, VIDEO_OUTPUT_DIR
    VIDEO_AVAILABLE = True
except Exception as e:
    print(f"❌ Error importing video_generator: {e}")
    print("⚠️ Video generation will be disabled")
    VIDEO_AVAILABLE = False
    VIDEO_OUTPUT_DIR = "video_output"

# Ensure required directories exist
os.makedirs("templates", exist_ok=True)

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

# -------------------------
# FLASK ROUTES
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        if not COMIC_AVAILABLE:
            return jsonify({
                'error': 'Comic generation not available due to dependency conflicts. Please fix your Python environment and restart.'
            }), 500
        
        # Extract request data
        data = request.get_json()
        story = data.get('story', '').strip()
        theme = data.get('theme', 'classic')
        
        if not story:
            return jsonify({'error': 'Please enter a story'}), 400
        
        # Validate theme
        if theme not in COMIC_THEMES:
            theme = 'classic'
        
        print(f"Generating comic for story: {story[:50]}... (Theme: {COMIC_THEMES[theme]['name']})")
        comic_paths, imgs_base64, num_panels, layout_info = generate_comic(story, theme)
        
        return jsonify({
            'success': True,
            'images': imgs_base64,  # Now an array of base64 images
            'paths': comic_paths,   # Array of file paths
            'theme': COMIC_THEMES[theme]['name'],
            'panels': num_panels,
            'pages': layout_info['pages'],
            'story_analysis': {
                'total_panels': num_panels,
                'total_pages': layout_info['pages'],
                'page_layouts': layout_info['layout_per_page']
            }
        })
    
    except Exception as e:
        print(f"Error generating comic: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/themes', methods=['GET'])
def get_themes():
    return jsonify({
        'themes': {key: value['name'] for key, value in COMIC_THEMES.items()}
    })

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/video_output/<filename>')
def video_output_file(filename):
    try:
        print(f"🎞️ Serving video file: {filename} from {VIDEO_OUTPUT_DIR}")
        video_path = os.path.join(VIDEO_OUTPUT_DIR, filename)
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            print(f"✅ Video file exists: {video_path} (Size: {file_size} bytes)")
            
            # Check if file is actually a video by trying to read it
            import cv2
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"📊 Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s duration")
            
            if frame_count == 0:
                print("❌ Video file has no frames - encoding issue")
                return "Video file is corrupted or has no content", 500
            
            # Create response with better headers for video streaming
            response = send_from_directory(VIDEO_OUTPUT_DIR, filename)
            response.headers['Content-Type'] = 'video/mp4'
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            response.headers['Content-Length'] = str(file_size)
            # Force download as fallback
            response.headers['Content-Disposition'] = 'inline; filename="' + filename + '"'
            print(f"📤 Serving video with headers: Content-Type=video/mp4, Size={file_size}")
            return response
        else:
            print(f"❌ Video file not found: {video_path}")
            return "Video file not found", 404
    except Exception as e:
        print(f"Error serving video: {e}")
        return f"Error serving video: {e}", 500

# Video generation Flask routes
@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        data = request.get_json()
        story = data.get('story', '').strip()
        style = data.get('style', 'cinematic')
        
        if not story:
            return jsonify({'error': 'Please enter a story'}), 400
        
        if len(story.split()) < 20:
            return jsonify({'error': 'Story too short. Please provide a longer story for video generation.'}), 400
        
        print(f"🎬 Generating video for story: {story[:50]}... (Style: {style})")
        
        result = generate_video_from_story(story, style)
        
        # Convert to relative path for web serving
        video_filename = os.path.basename(result['video_path'])
        
        # Debug info
        print(f"✅ Video generation complete!")
        print(f"📁 Video file: {result['video_path']}")
        print(f"📁 Video filename: {video_filename}")
        if os.path.exists(result['video_path']):
            file_size = os.path.getsize(result['video_path'])
            print(f"📊 File size: {file_size} bytes")
        else:
            print(f"⚠️ Warning: Video file not found at {result['video_path']}")
        
        # Debug: Print the returned story_analysis structure
        story_analysis = result['story_analysis']
        print(f"📊 Story analysis keys: {list(story_analysis.keys())}")
        
        return jsonify({
            'success': True,
            'video_path': f'/video_output/{video_filename}',
            'video_filename': video_filename,
            'num_scenes': result['total_scenes'],
            'story_analysis': {
                'scenes': len(story_analysis.get('scenes', [])),
                'summary': story_analysis.get('summary', 'Video generated with deep story understanding')
            },
            'duration_estimate': f"{result['total_scenes'] * 2} seconds",
            'style': result.get('style', style)
        })
        
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        return jsonify({'error': str(e)}), 500

# -------------------------
# CLI MODE (optional)
# -------------------------
def run_cli():
    from comic_generator import run_cli as comic_cli
    comic_cli()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'cli':
        run_cli()
    else:
        print("🎨 Comic & Video Generator Web Interface")
        print("📱 Open your browser and go to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)