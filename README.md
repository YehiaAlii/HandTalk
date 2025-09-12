# HandTalk 🤟
*Advanced Sign Language Translation System - Real-time Bidirectional Communication*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![WebSocket](https://img.shields.io/badge/WebSocket-Real--time-blue.svg)](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)

## Project Overview

HandTalk is a comprehensive sign language translation system that bridges the communication gap between deaf/hard-of-hearing individuals and hearing individuals. This graduation project implements a real-time, bidirectional translation system using state-of-the-art deep learning techniques with full mobile and web application support.

### What HandTalk Does:
- **Video-to-Text**: Converts American Sign Language (ASL) videos into English text
- **Text-to-Sign**: Generates sign language animations from English text input
- **Real-time Processing**: Provides instant translations through a web/mobile interface
- **Cross-Device Bridge**: Mobile and desktop devices communication
- **Multi-platform**: Works on web browsers, mobile devices, and desktop applications
- **Cloud-ready**: Runs entirely in Google Colab with free GPU access

### Technical Innovation:
- **Pose-only Recognition**: Uses efficient pose estimation instead of raw video processing
- **Uni-Sign Architecture**: Implements the latest sign language understanding model
- **WebSocket Communication**: Real-time bidirectional communication between users
- **RESTful API**: Complete backend system with multiple endpoints for different use cases
- **GPU Acceleration**: CUDA support with automatic CPU fallback

### Project Impact:
This system addresses the critical need for accessible communication tools in the deaf, hard-of-hearing community, and who rely on sign language as their primary means of communication.

## Platform Compatibility

### Mobile Applications
- **Real-time WebSocket communication** for instant translation updates
- **Cross-device synchronization** between mobile and desktop devices
- **Audio playback** for translated text with text-to-speech
- **Thumbnail previews** for quick video identification
- **Native file upload** with drag-and-drop support

### Web Applications  
- **RESTful API** with 15+ endpoints for complete functionality
- **WebSocket support** for real-time features and live updates
- **File upload with progress tracking** and visual feedback
- **Media endpoints** for video/audio/image delivery
- **Search functionality** with context highlighting

### Cross-Device Bridge System
- **Desktop → Mobile**: Video processing results automatically sent to mobile devices
- **Mobile → Desktop**: Text-to-sign requests generate animations on desktop
- **Real-time synchronization**: Live updates across all connected devices

**Professional Desktop Application**: [HandTalk-Bridge-Desktop](https://github.com/YehiaAlii/HandTalk-Bridge-Desktop)

*Complete desktop interface for institutional deployments with recording capabilities and real-time communication*

## Prerequisites & Setup

### Step 1: Get the Trained Model*
**This is REQUIRED before running the system.**

#### Option A: Train Your Own Model
1. **Clone the original Uni-Sign repository**:
   ```bash
   git clone https://github.com/ZechengLi19/Uni-Sign.git
   ```
2.  **Follow their training instructions** for OpenASL dataset:
    - See their README for detailed training steps
    - Requires significant GPU resources
    - Results in official_openasl_pose_only_slt.pth file

#### Option B: Obtain Pre-trained Weights
   - Get the trained model from someone who has already trained the model
   - Place the Model File:
```
HandTalk/out/official_checkpoints/official_openasl_pose_only_slt.pth
```
    
### Step 2: Get Pose Data for Text-to-Sign*
**REQUIRED for text-to-sign conversion feature**

1. **Download ASL pose data** from Google Drive:
```
https://drive.google.com/drive/folders/17oySRdzPl-7eX1-RqZubANVpG-uqnRtY
```

2. **Extract and place in:**
```
HandTalk/gloss2pose/asl/
└── [2000+ pose files]
```

### Step 3: Setup Google Drive Structure*
**Upload the complete HandTalk folder to your Google Drive in this exact structure:**

```
📁 Google Drive/
└── 📁 HandTalk/
    └── 📁 HandTalk/                    ← Repository folder
        ├── 📄 app.py
        ├── 📄 models.py
        ├── 📄 Inference.py
        ├── 📁 out/official_checkpoints/
        │   └── 📄 official_openasl_pose_only_slt.pth  ✅ **MODEL FILE**
        ├── 📁 gloss2pose/
        │   ├── 📄 words.txt
        │   └── 📁 asl/                  ✅ **POSE DATA**
        │       └── [2000+ pose files]
        ├── 📁 stgcn_layers/
        └── 📄 ... (all other files)
```

### Step 4: Ngrok Setup (For Public Access)*
**Required to make your Colab accessible from external devices**

1. **Create free ngrok account:**
   - Visit: https://ngrok.com/signup
   - Sign up and verify your email
2.  **Get your authentication token:**
   - Go to ngrok dashboard → "Your Authtoken"
   - Copy the token
3. **Add token to Google Colab Secrets:**
   - Open Google Colab
   - Click 🔑 Secrets in the left sidebar
   - Click + Add new secret
   - Name: NGROK_AUTH_TOKEN
   - Value: your_actual_token_here
   - Toggle Notebook access to ON

## Running the System

### Google Colab for free GPU

#### Step 1: Open Google Colab
1. **Go to**: https://colab.research.google.com/
2. **Sign in** with your Google account

#### Step 2: Setup Environment
**Copy and run this code block in Colab:**

```python
# Install all required packages
!pip install opencv-python tqdm fastapi uvicorn python-multipart gtts websockets pyngrok rtmlib onnxruntime-gpu bottle pose_format num2words scipy pillow pydantic torch torchvision transformers tokenizers sentencepiece numpy pandas matplotlib scikit-learn scikit-image rouge==1.0.1 sacrebleu einops timm decord accelerate deepspeed tensorboard tensorflow

# Install system dependencies
!apt-get update
!apt-get install -y ffmpeg
```

#### Step 3: Mount Google Drive & Load Project

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy project from Drive to Colab
!cp -r "/content/drive/My Drive/HandTalk/HandTalk" "/content/HandTalk"

# Navigate to project directory
%cd "/content/HandTalk"
```

#### Step 4: Start the HandTalk Server

```python
# Start FastAPI server in background
!nohup python -m uvicorn app:app --host 0.0.0.0 --port 8000 &

# Create public URL with ngrok
from pyngrok import ngrok
from google.colab import userdata

# Get ngrok token from Colab secrets
ngrok.set_auth_token(userdata.get("NGROK_AUTH_TOKEN"))
public_url = ngrok.connect(8000)

print(f"🌐 Public URL: {public_url}")
print(f"📚 API Documentation: {public_url.public_url}/docs")
print(f"🎯 Your HandTalk system is now live!")
```

#### Step Step 5: Monitor System Status

```python
# Check server logs for any errors
!tail -n 50 nohup.out
```

Your HandTalk system is now running and accessible at the public URL!


## API Documentation
*Complete API reference for mobile and web application integration*

### Base URL

```
{public_url} # Your ngrok URL from Colab setup
```


### Real-time Communication

#### WebSocket Connection
**Multi-device real-time communication system**
```javascript
// Connect with device identification
const ws = new WebSocket(`ws://{public_url}/ws/{device_type}/{device_id}`);
// device_type: "mobile" | "desktop" | "web"
// device_id: unique identifier for your device
```

### Upload Video for Translation
**Real-time video-to-text translation with WebSocket progress updates**

```http
POST /upload/
```

**Request:**
```bash
curl -X POST "{public_url}/upload/" \
  -F "file=@sign_video.mp4" \
  -F "client_id=unique_device_id" \
  -F "user_id=123" \
  -F "return_audio=true"
```

**Response:**
```json
{
  "translation_id": "1234567890",
  "message": "Processing started"
}
```

**WebSocket Progress Updates:**
```json
{
  "type": "progress",
  "translation_id": "1234567890",
  "status": "preprocessing|inference|completed|error",
  "message": "Running sign language translation model...",
  "progress": 50,
  "text": "Hello world"
}
```

### Cross-Device Video Upload
```http
POST /upload-device/
```
**Automatically broadcasts translation results to all connected mobile devices**

### Get Translation Result
```http
GET /result/{translation_id}?return_audio=true
```

**Response:**
```json
{
  "translation_id": "1234567890",
  "text": "Hello world",
  "user_id": "123",
  "original_filename": "my_video.mp4",
  "audio_url": "{public_url}/audio/temp/1234567890"
}
```

### Save V2T Translation
```http
POST /save/v2t
```

**Request:**
```json
{
  "translation_id": "1234567890",
  "user_id": 123,
  "return_audio": true
}
```

## Text-to-Sign (T2V) APIs

### Convert Text to Sign Language
```http
GET /gloss2pose?gloss=hello world&user_id=123
```

**Response:**
```json
{
  "img": "base64_encoded_gif_data",
  "words": ["HELLO", "WORLD"],
  "processing_time": 2.1,
  "word_count": 2,
  "original_text": "hello world",
  "isSaved": false,
  "translation_id": null
}
```

### Save T2V Translation
```http
POST /t2v/save
```

**Request:**
```json
{
  "user_id": 123,
  "text": "hello world",
  "generate_audio": true
}
```

**Response:**
```json
{
  "translation_id": 1,
  "message": "Translation saved successfully",
  "has_thumbnail": true,
  "audio_url": "{public_url}/user/123/t2v/audio/1",
  "saved_at": "2024-01-15 10:30:00"
}
```

### Get All T2V Translations
```http
GET /retrieve_translations?user_id=123
```

### Delete T2V Translation
```http
DELETE /delete_translations/{translation_id}?user_id=123
```

## User Data Management APIs

### Recent Items (Last 3)
```http
GET /user/{user_id}/v2t/recent-translations       # Recent V2T translations
GET /user/{user_id}/t2v/recent-translations       # Recent T2V translations
GET /user/{user_id}/recent-translations/all       # Combined recent from both systems

GET /user/{user_id}/v2t/recent-saves              # Recent V2T saves
GET /user/{user_id}/t2v/recent-saves              # Recent T2V saves  
GET /user/{user_id}/recent-saves/all              # Combined recent saves
```

### All Saved Translations
```http
GET /user/{user_id}/translations/with-text        # V2T text-based saves
GET /user/{user_id}/translations/with-audio       # V2T audio-enabled saves
GET /user/{user_id}/saved/all                     # All saved (V2T + T2V combined)
```

**Response Example:**
```json
{
  "translations": [
    {
      "id": "123",
      "text": "Hello world",
      "timestamp": "2024-01-15T10:30:00Z",
      "filename": "my_video.mp4",
      "source_type": "video-to-text|text-to-sign",
      "has_audio": true,
      "video_url": "{public_url}/user/123/translations/video/123",
      "audio_url": "{public_url}/user/123/translations/audio/123",
      "thumbnail_url": "{public_url}/user/123/translations/thumbnail/123"
    }
  ],
  "count": 1
}
```

### Delete Saved Translations
```http
DELETE /user/{user_id}/translations/with-text/{translation_id}    # Delete V2T text
DELETE /user/{user_id}/translations/with-audio/{translation_id}   # Delete V2T audio
```

## Search APIs

```http
GET /user/{user_id}/search?query=hello            # Search all translations
GET /user/{user_id}/search/v2t?query=hello        # Search V2T translations only
GET /user/{user_id}/search/t2v?query=hello        # Search T2V translations only
```

**Response with context highlighting:**
```json
{
  "results": [
    {
      "id": "123",
      "text": "Hello world, how are you?",
      "match_context": "...world, <highlight>hello</highlight> there...",
      "source_type": "video-to-text|text-to-sign",
      "type": "text|audio|animation",
      "has_audio": true,
      "has_thumbnail": true,
      "video_url": "{public_url}/user/123/translations/video/123",
      "audio_url": "{public_url}/user/123/translations/audio/123",
      "thumbnail_url": "{public_url}/user/123/translations/thumbnail/123",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "query": "hello",
  "sources": {
    "video_to_text": 1,
    "text_to_sign": 0
  }
}
```

**Search Features:**
- **Case-insensitive** search across all saved translations
- **Context highlighting** with `<highlight>` tags for easy UI integration  
- **Multi-source** search across both V2T and T2V systems
- **Result ranking** by timestamp (newest first)

## Media File APIs

### Direct Media Access
```http
# V2T Media Files
GET /user/{user_id}/translations/video/{translation_id}           # Video files
GET /user/{user_id}/translations/audio/{translation_id}           # Audio files
GET /user/{user_id}/translations/thumbnail/{translation_id}       # Video thumbnails

# T2V Media Files  
GET /user/{user_id}/t2v/thumbnail/{translation_id}               # T2V thumbnails
GET /user/{user_id}/t2v/audio/{translation_id}                   # T2V audio files

# Recent Media Files
GET /user/{user_id}/recent-translations/v2t/thumbnail/{thumbnail_id}    # Recent V2T thumbnails
GET /user/{user_id}/recent-translations/t2v/thumbnail/{thumbnail_id}    # Recent T2V thumbnails
GET /user/{user_id}/recent-translations/audio/{translation_id}          # Recent audio files
```

**Media Response Headers:**
```http
Content-Type: video/mp4                    # For video files
Content-Type: audio/mpeg                   # For audio files  
Content-Type: image/jpeg                   # For thumbnail files
Content-Disposition: inline; filename="123.mp4"
```


**Media Features:**
- **Direct streaming** - Files served with proper headers for browser playback
- **Thumbnail previews** - Auto-generated from videos (frame 5) and GIFs (middle frame)
- **Audio generation** - Text-to-speech MP3 files with high quality

# Bridge Communication APIs

### Cross-Device Conversation History
```http
GET /bridge-conversation                    # Full conversation history
GET /bridge-conversation/recent            # Last 2 messages only
```

**Response:**
```json
{
  "title": "Sign Language Bridge Conversation",
  "messages": [
    {
      "role": "user|assistant",
      "content": "Hello world",
      "timestamp": "2024-01-15T10:30:00Z",
      "type": "text_to_sign|video_to_text"
    }
  ],
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```
