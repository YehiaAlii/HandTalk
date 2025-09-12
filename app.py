from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect, Request, Query, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
import io
import gtts  # For text-to-speech
import json
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from typing import Dict, List, Optional
import pathlib
import datetime
from datetime import datetime
from bottle import route, run, request, response, hook, error
import socket
from pose_format.pose_visualizer import PoseVisualizer
from pydantic import BaseModel
from typing import Union
import time
import cv2
import warnings
import base64
import logging
from collections import OrderedDict
from typing import Optional, Tuple
from Inference import translate_sign_language, InferenceConfig, get_model
from preprocessing import extract_keypoints_from_video
import glosstopose
from PIL import Image
import torch

from fastapi.middleware.cors import CORSMiddleware

import logging

from logging.handlers import RotatingFileHandler

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            "server.log",  # Log file name
            maxBytes=10485760,  # 10 MB per file
            backupCount=5,  # Keep 5 backup files
            encoding='utf-8'
        )
    ]
)

logger = logging.getLogger(__name__)


# Define storage root path
STORAGE_ROOT = pathlib.Path("./storage")


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.device_info: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, device_id: str, device_type: str):
        await websocket.accept()
        self.active_connections[device_id] = websocket
        self.device_info[device_id] = {
            "type": device_type,
            "websocket": websocket
        }
        print(f"Device {device_id} ({device_type}) connected")
        
        # Notify all clients about the new connection
        await self.broadcast_system_message(f"{device_type} device {device_id} connected", device_id)
    
    def disconnect(self, device_id: str):
        device_type = self.device_info.get(device_id, {}).get("type", "unknown")
        if device_id in self.active_connections:
            del self.active_connections[device_id]
            del self.device_info[device_id]
            print(f"Device {device_id} disconnected")
    
    async def send_personal_message(self, message: str, device_id: str):
        if device_id in self.active_connections:
            websocket = self.active_connections[device_id]
            try:
                await websocket.send_text(message)
            except Exception as e:
                print(f"Error sending message to {device_id}: {e}")
                # Remove the broken connection
                self.disconnect(device_id)
    
    async def broadcast_message(self, message: str, sender_id: str = None):
        disconnected_devices = []
        
        for device_id, websocket in self.active_connections.items():
            if device_id != sender_id:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    print(f"Error broadcasting to {device_id}: {e}")
                    disconnected_devices.append(device_id)
        
        # Clean up disconnected devices
        for device_id in disconnected_devices:
            self.disconnect(device_id)
    
    async def broadcast_system_message(self, message: str, exclude_id: str = None):
        system_message = {
            "type": "system",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast_message(json.dumps(system_message), exclude_id)


manager = ConnectionManager()


def save_bridge_message_single(content, role, timestamp, message_type):
    """Save a single bridge conversation message to a JSON file"""
    # Create bridge directory if it doesn't exist
    bridge_dir = "bridge_conversations"
    os.makedirs(bridge_dir, exist_ok=True)
    
    # Path to the bridge conversation file
    bridge_file = os.path.join(bridge_dir, "conversation.json")
    
    # Check if file exists
    if os.path.exists(bridge_file):
        # Read existing conversation data
        with open(bridge_file, 'r') as f:
            conversation_data = json.load(f)
        
        # Add new message
        conversation_data['messages'].append({
            'role': role,
            'content': content,
            'timestamp': timestamp,
            'type': message_type
        })
        
        # Update timestamp
        conversation_data['updated_at'] = timestamp
    else:
        # Create new conversation data
        conversation_data = {
            'title': 'Sign Language Bridge Conversation',
            'messages': [
                {
                    'role': role,
                    'content': content,
                    'timestamp': timestamp,
                    'type': message_type
                }
            ],
            'created_at': timestamp,
            'updated_at': timestamp
        }
    
    # Write conversation data to file
    with open(bridge_file, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    
    print(f"Saved bridge conversation: {message_type} - {role}")



async def process_video_device(video_path: str, client_id: str, original_filename: str):
    """
    Process video file and send result to mobile device
    """
    try:
        # Send processing started notification
        if client_id in manager.active_connections:
            await manager.send_personal_message(
                json.dumps({
                    "type": "video_processing",
                    "status": "started",
                    "message": f"Processing video: {original_filename}",
                    "timestamp": datetime.now().isoformat()
                }),
                client_id
            )
        
        # Use your existing inference logic
        try:
            # Extract keypoints from video
            pose_data = extract_keypoints_from_video(
                video_path,
                device='cuda',
                backend='onnxruntime',
                max_workers=4
            )

            config = InferenceConfig()

            translated_text = translate_sign_language(video_path, config, pre_extracted_keypoints=pose_data)
            
            if not translated_text:
                translated_text = "No translation could be generated for this video"
                
        except Exception as inference_error:
            print(f"Inference error: {inference_error}")
            translated_text = f"Error during translation: {str(inference_error)}"
        

        timestamp = datetime.now().isoformat()

        save_bridge_message_single(
            content=translated_text,
            role="assistant",
            timestamp=timestamp,
            message_type="video_to_text"
        )

        # Send result to mobile devices (broadcast to all mobile devices)
        translation_message = {
            "type": "translation_result",
            "from": client_id,
            "device_type": "desktop",
            "message": translated_text,
            "original_filename": original_filename,
            "timestamp": timestamp
        }

        
        # Send to all mobile devices
        for device_id, info in manager.device_info.items():
            if info["type"] == "mobile":
                await manager.send_personal_message(
                    json.dumps(translation_message),
                    device_id
                )
        
        # Send completion notification to desktop
        if client_id in manager.active_connections:
            await manager.send_personal_message(
                json.dumps({
                    "type": "video_processing",
                    "status": "completed",
                    "message": f"Video processed successfully: {original_filename}",
                    "timestamp": datetime.now().isoformat()
                }),
                client_id
            )
            
    except Exception as e:
        # Send error notification
        if client_id in manager.active_connections:
            await manager.send_personal_message(
                json.dumps({
                    "type": "video_processing",
                    "status": "error",
                    "message": f"Error processing video: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }),
                client_id
            )
    finally:
        # Clean up temporary file
        if os.path.exists(video_path):
            os.unlink(video_path)


async def process_text_to_sign(text: str, client_id: str, device_type: str):
    """
    Process text to sign language GIF and send result to desktop devices
    """
    try:
        # Send processing started notification
        if client_id in manager.active_connections:
            await manager.send_personal_message(
                json.dumps({
                    "type": "text_processing",
                    "status": "started",
                    "message": f"Converting text to sign language: {text}",
                    "timestamp": datetime.now().isoformat()
                }),
                client_id
            )
        
        # Process text to sign language (based on your existing logic)
        start = time.time()
        text_normalized = text.strip().lower()
        
        # Prepare glosses
        glosses = glosstopose.prepare_glosses(text_normalized)
        if not glosses:
            raise Exception(f"No gloss found for: {text}")
        
        # Get pose data
        pose, words = POSE_LOOKUP.gloss_to_pose(glosses)
        if not pose:
            raise Exception(f"No pose found for words: {', '.join(glosses)}")
        
        # Generate visualization
        glosstopose.scale_down(pose, 512)
        p = PoseVisualizer(pose, thickness=2)
        
        # Use CUDA if available
        if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
            try:
                with torch.cuda.amp.autocast():
                    img = p.save_gif(None, p.draw())
            except Exception as e:
                print(f"Failed to use GPU for visualization: {e}, falling back to CPU")
                img = p.save_gif(None, p.draw())
        else:
            img = p.save_gif(None, p.draw())
        
        # Encode to base64
        img_base64 = base64.b64encode(img).decode('utf-8')
        processing_time = time.time() - start
        
        # Create response data
        response_data = {
            "img": img_base64,
            "words": words,
            "processing_time": processing_time,
            "word_count": len(words),
            "original_text": text,
            "isSaved": False,
            "translation_id": None
        }
        

        timestamp = datetime.now().isoformat()

        # Send result to desktop devices
        save_bridge_message_single(
            content=text,
            role="user",
            timestamp=timestamp,
            message_type="text_to_sign"
        )
        
        sign_language_message = {
            "type": "sign_language_result",
            "from": client_id,
            "device_type": device_type,
            "data": response_data,
            "timestamp": timestamp
        }

        # Send to all desktop devices
        for device_id, info in manager.device_info.items():
            if info["type"] == "desktop":
                await manager.send_personal_message(
                    json.dumps(sign_language_message),
                    device_id
                )
        
        # Send completion notification to mobile
        if client_id in manager.active_connections:
            await manager.send_personal_message(
                json.dumps({
                    "type": "text_processing",
                    "status": "completed",
                    "message": f"Text converted to sign language successfully",
                    "timestamp": datetime.now().isoformat()
                }),
                client_id
            )
            
    except Exception as e:
        # Send error notification
        if client_id in manager.active_connections:
            await manager.send_personal_message(
                json.dumps({
                    "type": "text_processing",
                    "status": "error",
                    "message": f"Error converting text: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }),
                client_id
            )
        
        # Also send error to desktop devices
        error_message = {
            "type": "sign_language_error",
            "from": client_id,
            "device_type": device_type,
            "error": str(e),
            "original_text": text,
            "timestamp": datetime.now().isoformat()
        }
        
        for device_id, info in manager.device_info.items():
            if info["type"] == "desktop":
                await manager.send_personal_message(
                    json.dumps(error_message),
                    device_id
                )

app = FastAPI(
    title="HandTalk API",
    description="API for translating sign language videos to text and converting text to sign language",
    version="1.0.0"
)

# Enable CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize model at startup"""
    print("Preloading translation model...")
    config = InferenceConfig()
    get_model(config)
    print("Model preloaded successfully")

# Initialize the POSE_LOOKUP object
try:
    POSE_LOOKUP = glosstopose.PoseLookup("gloss2pose", "asl")
    warnings.filterwarnings("ignore")
except Exception as e:
    print(f"Failed to initialize POSE_LOOKUP: {e}")
    raise


# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Store translation results
translation_results: Dict[str, Dict] = {}

# Add a root endpoint
@app.get("/")
async def root():
    """API information and available endpoints"""
    return {
        "message": "Sign Language Translator API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "websocket": "/ws/{client_id}",
            "upload": "/upload/",
            "result": "/result/{translation_id}",
            "save": "/save/",
            "translations_with_text": "/user/{user_id}/translations/with-text",
            "translations_with_audio": "/user/{user_id}/translations/with-audio",
            "translation_audio": "/user/{user_id}/translations/audio/{translation_id}"
        }
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections[client_id] = websocket

    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    # Respond to ping with pong
                    await websocket.send_json({"type": "pong"})
            except:
                pass

    except WebSocketDisconnect:
        # Remove connection when client disconnects
        if client_id in active_connections:
            del active_connections[client_id]

@app.post("/upload/")
async def upload_video(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    user_id: str = Form(...),
    return_audio: bool = Form(False),
    request: Request = None
):
    """
    Upload a video file for translation

    Args:
        file: The video file to translate
        client_id: ID of the WebSocket connection
        user_id: ID of the user
        return_audio: Whether to return audio instead of text

    Returns:
        A translation ID that can be used to get the final result
    """
    # Check if client is connected via WebSocket
    if client_id not in active_connections:
        raise HTTPException(
            status_code=400,
            detail="Client not connected via WebSocket. Connect to /ws/{client_id} first"
        )

    try:
      user_id_int = int(user_id)
    except ValueError:
      raise HTTPException(
        status_code=400,
        detail="User ID must be convertible to an integer" )

    translation_id = str(get_next_id(user_id_int))

    # Create a temporary file for the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        # Write the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()

        original_filename = file.filename

        # Start processing in a background task
        asyncio.create_task(
            process_video(
                translation_id=translation_id,
                client_id=client_id,
                video_path=temp_file.name,
                user_id=user_id,
                return_audio=return_audio,
                original_filename=original_filename,
                request=request
            )
        )

        # Return translation ID to client
        return {"translation_id": translation_id, "message": "Processing started"}

    except Exception as e:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

        # Send error notification via WebSocket
        try:
            websocket = active_connections[client_id]
            await websocket.send_json({
                "type": "error",
                "message": f"Error processing upload: {str(e)}"
            })
        except:
            pass

        # Re-raise the exception
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )

@app.get("/result/{translation_id}")
async def get_result(translation_id: str, return_audio: bool = False, request: Request = None):
    """
    Get the final result for a completed translation

    Args:
        translation_id: ID of the translation
        return_audio: Whether to return audio URL instead of just text
        request: Request object to get base URL

    Returns:
        Text result with optional audio URL
    """
    if translation_id not in translation_results:
        raise HTTPException(
            status_code=404,
            detail=f"Translation {translation_id} not found"
        )

    result = translation_results[translation_id]

    if result["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Translation {translation_id} is not completed yet"
        )

    response = {
        "translation_id": translation_id,
        "text": result["text"],
        "user_id": result["user_id"],
        "original_filename": result.get("original_filename", "unknown")
    }

    if return_audio and request:
        try:
            # Generate the audio and save it temporarily
            audio_dir = STORAGE_ROOT / "temp_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{translation_id}.mp3"
            
            # Only generate if it doesn't exist
            if not audio_path.exists():
                tts = gtts.gTTS(result["text"])
                tts.save(str(audio_path))
            
            # Add audio URL to the response
            base_url = str(request.base_url) if request else ""
            response["audio_url"] = f"{base_url}audio/temp/{translation_id}"
            
        except Exception as e:
            # Just log the error but don't fail the request
            print(f"Error generating audio: {str(e)}")
    
    return response

# Endpoint to serve the temporary audio files:
@app.get("/audio/temp/{audio_id}")
async def get_temp_audio(audio_id: str):
    """
    Get a temporary audio file
    
    Args:
        audio_id: ID of the audio file
        
    Returns:
        Audio file
    """
    audio_path = STORAGE_ROOT / "temp_audio" / f"{audio_id}.mp3"
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    
    # Return the audio file with inline disposition for playing in browser
    return Response(
        content=open(audio_path, "rb").read(),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"inline; filename={audio_id}.mp3"}
    )

async def process_video(translation_id: str, client_id: str, video_path: str, user_id: str, return_audio: bool, original_filename: str, request: Request = None):
    """
    Process the video in the background and send progress updates via WebSocket

    Args:
        translation_id: ID of the translation translation
        client_id: ID of the WebSocket connection
        video_path: Path to the video file
        user_id: ID of the user
        return_audio: Whether audio was requested
        original_filename: Original filename of the uploaded video
    """
    websocket = active_connections.get(client_id)
    if not websocket:
        # Client disconnected
        return

    # Initialize translation status
    translation_results[translation_id] = {
        "status": "processing",
        "text": None,
        "user_id": user_id,
        "video_path": video_path,
        "original_filename": original_filename
    }

    try:
        # Send initial notification
        await websocket.send_json({
            "type": "progress",
            "translation_id": translation_id,
            "status": "started",
            "message": "Translation process started",
            "progress": 0
        })

        # Step 1: Extract keypoints (preprocessing)
        await websocket.send_json({
            "type": "progress",
            "translation_id": translation_id,
            "status": "preprocessing",
            "message": "Extracting keypoints from video...",
            "progress": 10
        })

        # Extract keypoints
        pose_data = extract_keypoints_from_video(
            video_path,
            device='cuda',
            backend='onnxruntime',
            max_workers=4
        )

        # Step 2: Run the translation model (inference)
        await websocket.send_json({
            "type": "progress",
            "translation_id": translation_id,
            "status": "inference",
            "message": "Running sign language translation model...",
            "progress": 50
        })

        # Create config and run translation
        config = InferenceConfig()
        text_result = translate_sign_language(video_path, config, pre_extracted_keypoints=pose_data)

        # Update result
        translation_results[translation_id] = {
            "status": "completed",
            "text": text_result,
            "user_id": user_id,
            "video_path": video_path,
            "original_filename": original_filename
        }

        try:
            user_id_int = int(user_id)
            await update_recent_v2t_translations(
                user_id=user_id_int,
                save_id=translation_id,
                text=text_result,
                video_path=video_path,
                generate_audio=return_audio,
                original_filename=original_filename,
                request=request
            )
            print(f"Updated recent translations for user {user_id_int} with translation {translation_id}")
        except Exception as e:
            print(f"Error updating recent translations: {e}")
            import traceback
            traceback.print_exc()

        # Send completion notification
        await websocket.send_json({
            "type": "progress",
            "translation_id": translation_id,
            "status": "completed",
            "message": "Translation completed successfully",
            "progress": 100,
            "text": text_result,
            "original_filename": original_filename
        })

        # Generate audio if requested
        if return_audio:
            audio_dir = STORAGE_ROOT / "temp_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{translation_id}.mp3"

            try:
                if not audio_path.exists():
                    tts = gtts.gTTS(text_result)
                    tts.save(str(audio_path))

                if request:
                    base_url = str(request.base_url)
                    audio_url = f"{base_url}audio/temp/{translation_id}"                

                    await websocket.send_json({
                        "type": "progress",
                        "translation_id": translation_id,
                        "status": "audio_ready",
                        "message": "Audio is ready for playback",
                        "audio_url": audio_url
                    })
                else:
                    print("Request object not available, skipping audio URL in WebSocket notification")

            except Exception as e:
                print(f"Error generating audio: {e}")
                await websocket.send_json({
                    "type": "warning",
                    "translation_id": translation_id,
                    "status": "audio_failed",
                    "message": f"Audio generation failed: {str(e)}"
                     })

    except Exception as e:
        # Update status with error
        translation_results[translation_id] = {
            "status": "failed",
            "text": None,
            "error": str(e),
            "user_id": user_id,
            "original_filename": original_filename
        }

        # Send error notification
        if websocket:
            await websocket.send_json({
                "type": "error",
                "translation_id": translation_id,
                "message": f"Error during translation: {str(e)}"
            })

        # Log the error
        import traceback
        traceback.print_exc()

    finally:

        if translation_id not in translation_results or translation_results[translation_id]["status"] != "completed":
          if os.path.exists(video_path):
            os.unlink(video_path)

def extract_frame(video_path, frame_number, output_path):
    """Extract a specific frame from a video and save it as an image

    Args:
        video_path: Path to the video file
        frame_number: Frame number to extract (0-based)
        output_path: Path where the frame will be saved

    Returns:
        True if successful, False otherwise
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))

        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False

        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if frame_number is valid
        if frame_number >= total_frames:
            print(f"Error: Frame {frame_number} does not exist in video with {total_frames} frames")
            return False

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        # Check if frame was successfully read
        if not ret:
            print(f"Error: Could not read frame {frame_number}")
            return False

        # Save the frame as an image
        cv2.imwrite(str(output_path), frame)

        # Release the video capture object
        cap.release()

        return True
    except Exception as e:
        print(f"Error extracting frame: {str(e)}")
        return False


async def update_recent_v2t_translations(user_id: int, save_id: str, text: str, video_path: str, generate_audio: bool, original_filename: str = "unknown", request: Request = None):
    """Update the recent translations for a user"""
    try:
        print(f"Starting update_recent_v2t_translations for user {user_id}, save_id {save_id}")

        # Get base URL from request if available
        base_url = str(request.base_url) if request else ""

        # Log actual paths to verify they're correct
        user_dir = STORAGE_ROOT / "users" / str(user_id)
        recent_dir = user_dir / "recent_translations"
        print(f"User directory: {user_dir} (exists: {user_dir.exists()})")
        print(f"Recent translations directory: {recent_dir}")

        # Create directories with verbose logging
        videos_dir = recent_dir / "videos"
        translations_dir = recent_dir / "translations"
        audio_dir = recent_dir / "audio"
        metadata_dir = recent_dir / "metadata"
        thumbnails_dir = recent_dir / "thumbnails"

        try:
            videos_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created videos directory: {videos_dir} (exists: {videos_dir.exists()})")
        except Exception as e:
            print(f"Error creating videos directory: {e}")

        try:
            translations_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created translations directory: {translations_dir} (exists: {translations_dir.exists()})")
        except Exception as e:
            print(f"Error creating translations directory: {e}")

        try:
            audio_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created audio directory: {audio_dir} (exists: {audio_dir.exists()})")
        except Exception as e:
            print(f"Error creating audio directory: {e}")

        try:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created metadata directory: {metadata_dir} (exists: {metadata_dir.exists()})")
        except Exception as e:
            print(f"Error creating metadata directory: {e}")

        try:
            thumbnails_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created thumbnails directory: {thumbnails_dir} (exists: {thumbnails_dir.exists()})")
        except Exception as e:
            print(f"Error creating thumbnails directory: {e}")

        video_dest = videos_dir / f"{save_id}.mp4"
        try:
            print(f"Source video path: {video_path} (exists: {os.path.exists(video_path)})")
            shutil.copy2(video_path, video_dest)
            print(f"Copied video to: {video_dest} (exists: {video_dest.exists()})")

            # Create thumbnail from video
            has_thumbnail = False
            thumbnail_url = None
            try:
                thumbnail_path = thumbnails_dir / f"{save_id}.jpg"
                has_thumbnail = extract_frame(video_dest, 5, thumbnail_path)  # Extract the 5th frame as thumbnail
                print(f"Created thumbnail: {thumbnail_path} (exists: {thumbnail_path.exists()}, success: {has_thumbnail})")

                if has_thumbnail and base_url:
                    thumbnail_url = f"{base_url}user/{user_id}/recent-translations/v2t/thumbnail/{save_id}"
                elif has_thumbnail:
                    thumbnail_url = f"/user/{user_id}/recent-translations/v2t/thumbnail/{save_id}"
            except Exception as e:
                print(f"Error creating thumbnail: {e}")
                has_thumbnail = False

        except Exception as e:
            print(f"Error copying video file: {e}")
            has_thumbnail = False

        # Create metadata file
        metadata_path = metadata_dir / f"{save_id}.json"
        try:
            audio_url = None
            if generate_audio:
                if base_url:
                    audio_url = f"{base_url}user/{user_id}/recent-translations/audio/{save_id}"
                else:
                    audio_url = f"/user/{user_id}/recent-translations/audio/{save_id}"

            metadata_content = {
                "original_filename": original_filename,
                "translation_id": save_id,
                "timestamp": datetime.now().isoformat(),
                "has_audio": generate_audio,
                "has_thumbnail": has_thumbnail,
                "source_type": "video-to-text"
            }

            if has_thumbnail and thumbnail_url:
                metadata_content["thumbnail_url"] = thumbnail_url

            if generate_audio and audio_url:
                metadata_content["audio_url"] = audio_url

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_content, f)
            print(f"Created metadata file: {metadata_path} (exists: {metadata_path.exists()})")
        except Exception as e:
            print(f"Error creating metadata file: {e}")

        # Create translation file
        translation_path = translations_dir / f"{save_id}.txt"
        try:
            with open(translation_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Created translation file: {translation_path} (exists: {translation_path.exists()})")
        except Exception as e:
            print(f"Error creating translation file: {e}")

        # Generate audio if requested
        if generate_audio:
            try:
                audio_path = audio_dir / f"{save_id}.mp3"
                text_path = audio_dir / f"{save_id}.txt"

                # Generate audio
                tts = gtts.gTTS(text)
                tts.save(str(audio_path))
                print(f"Created audio file: {audio_path} (exists: {audio_path.exists()})")

                # Save duplicate text
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Created audio text file: {text_path} (exists: {text_path.exists()})")
            except Exception as e:
                print(f"Error generating audio: {e}")

                # Update metadata to reflect that audio generation failed
                try:
                    metadata_content["has_audio"] = False
                    if "audio_url" in metadata_content:
                        del metadata_content["audio_url"]
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata_content, f)
                except:
                    pass

        # Update recent_translations.json
        recent_json_path = user_dir / "recent_translations.json"
        try:
            recent_list = []

            # Read existing file if it exists
            if recent_json_path.exists():
                try:
                    with open(recent_json_path, "r") as f:
                        recent_list = json.load(f)
                    print(f"Read existing recent_translations.json with {len(recent_list)} items")
                except json.JSONDecodeError:
                    print("recent_translations.json exists but contains invalid JSON")
                    recent_list = []

            # Add new translation
            new_item = {
                "id": save_id,
                "timestamp": datetime.now().isoformat(),
                "filename": original_filename,
                "has_thumbnail": has_thumbnail,
                "has_audio": generate_audio,
                "text": text,
                "source_type": "video-to-text"
            }

            if has_thumbnail and thumbnail_url:
                new_item["thumbnail_url"] = thumbnail_url

            if generate_audio and audio_url:
                new_item["audio_url"] = audio_url

            recent_list.insert(0, new_item)
            print(f"Added new translation to list, now has {len(recent_list)} items")

            # Keep only 3 most recent
            if len(recent_list) > 3:
                ids_to_remove = [item["id"] for item in recent_list[3:]]
                recent_list = recent_list[:3]
                print(f"Trimmed list to 3 items, will remove old files with IDs: {ids_to_remove}")

                # Remove files for old translations
                for old_id in ids_to_remove:
                    try:
                        old_files = [
                            (videos_dir / f"{old_id}.mp4", "video"),
                            (translations_dir / f"{old_id}.txt", "translation"),
                            (audio_dir / f"{old_id}.mp3", "audio"),
                            (audio_dir / f"{old_id}.txt", "audio text"),
                            (metadata_dir / f"{old_id}.json", "metadata"),
                            (thumbnails_dir / f"{old_id}.jpg", "thumbnail")
                        ]

                        for old_file, file_type in old_files:
                            if old_file.exists():
                                os.unlink(old_file)
                                print(f"Removed old {file_type} file: {old_file}")
                    except Exception as e:
                        print(f"Error removing old files for ID {old_id}: {e}")

            # Write updated list
            with open(recent_json_path, "w") as f:
                json.dump(recent_list, f)
            print(f"Updated recent_translations.json at {recent_json_path}")

        except Exception as e:
            print(f"Error updating recent_translations.json: {e}")

        print("Successfully completed update_recent_v2t_translations")

    except Exception as e:
        print(f"Error in update_recent_v2t_translations: {e}")
        import traceback
        traceback.print_exc()



# Keep only this function for generating sequential IDs
def get_next_id(user_id: int) -> int:
    """Get the next sequential ID for a user's saved translations"""
    user_dir = STORAGE_ROOT / "users" / str(user_id)
    counter_file = user_dir / "id_counter.json"

    # Create user directory if it doesn't exist
    user_dir.mkdir(parents=True, exist_ok=True)

    # Read current counter or initialize to 1
    if counter_file.exists():
        try:
            with open(counter_file, "r") as f:
                counter_data = json.load(f)
                current_counter = counter_data.get("next_id", 1)
        except (json.JSONDecodeError, KeyError):
            current_counter = 1
    else:
        current_counter = 1

    # Update counter for next use
    with open(counter_file, "w") as f:
        json.dump({"next_id": current_counter + 1}, f)

    return current_counter

# Define a model for the save request
class SaveRequest(BaseModel):
    translation_id: str
    user_id: Union[int, str]
    return_audio: bool = False

@app.post("/save/v2t")
async def save_v2t_translation(request: SaveRequest):
    """
    Save a completed translation to the user's saved videos folder

    Args:
        translation_id: ID of the completed translation translation
        user_id: ID of the user (will be converted to int)
        return_audio: Whether to save audio as well

    Returns:
        Success message and saved ID
    """
    # Convert user_id to int if it's not already
    try:
        user_id = int(request.user_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="User ID must be convertible to an integer"
        )

    # Check if the translation exists and is completed
    if request.translation_id not in translation_results:
        raise HTTPException(
            status_code=404,
            detail=f"Translation {request.translation_id } not found"
        )

    result = translation_results[request.translation_id]

    if result["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"translation {request.translation_id} is not completed yet"
        )

    # Generate a unique ID for the saved translation
    # Using a simple timestamp-based ID for simplicity
    save_id = request.translation_id

    # Create the directory structure
    user_dir = STORAGE_ROOT / "users" / str(user_id)
    saved_videos_dir = user_dir / "saved" / "videos"
    saved_translations_dir = user_dir / "saved" / "translations"
    saved_audio_dir = user_dir / "saved" / "audio"
    saved_metadata_dir = user_dir / "saved" / "metadata"
    saved_thumbnails_dir = user_dir / "saved" / "thumbnails"

    # Create directories if they don't exist
    saved_videos_dir.mkdir(parents=True, exist_ok=True)
    saved_translations_dir.mkdir(parents=True, exist_ok=True)
    saved_audio_dir.mkdir(parents=True, exist_ok=True)
    saved_metadata_dir.mkdir(parents=True, exist_ok=True)
    saved_thumbnails_dir.mkdir(parents=True, exist_ok=True)

    # Get paths for the original temporary files
    temp_video_path = result.get("video_path")
    if not temp_video_path or not os.path.exists(temp_video_path):
        # The original video might have been deleted, so we can't save it
        raise HTTPException(
            status_code=400,
            detail="Original video is no longer available"
        )

    # Save the video
    video_path = saved_videos_dir / f"{save_id}.mp4"
    try:
        # Copy the video file
        shutil.copy2(temp_video_path, video_path)
        thumbnail_path = saved_thumbnails_dir / f"{save_id}.jpg"
        extract_frame(video_path, 5, thumbnail_path)

        metadata_path = saved_metadata_dir / f"{save_id}.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
          json.dump({
              "original_filename": result.get("original_filename", "unknown"),
              "translation_id": save_id,
              "timestamp": datetime.now().isoformat(),
              "has_audio": request.return_audio
              }, f)

        # If audio was requested, generate and save it
        if request.return_audio:
            audio_path = saved_audio_dir / f"{save_id}.mp3"
            text_path = saved_audio_dir / f"{save_id}.txt"

            # Generate audio
            tts = gtts.gTTS(result["text"])
            tts.save(str(audio_path))

            # Save duplicate text for searchability
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

        else:
          translation_path = saved_translations_dir / f"{save_id}.txt"
          with open(translation_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        # Update recent_saves.json
        recent_saves_path = user_dir / "recent_saves.json"
        recent_saves = []

        # Read existing recent saves if file exists
        if recent_saves_path.exists():
            try:
                with open(recent_saves_path, "r") as f:
                    recent_saves = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start with empty list
                recent_saves = []

        # Add new save to the beginning
        recent_saves.insert(0, {
            "id": save_id,
            "timestamp": datetime.now().isoformat(),
            "filename": result.get("original_filename", "unknown")
        })

        # Keep only the 3 most recent
        recent_saves = recent_saves[:3]

        # Write back to file
        with open(recent_saves_path, "w") as f:
            json.dump(recent_saves, f)

        return {
            "success": True,
            "message": "Translation saved successfully",
            "id": save_id
        }

    except Exception as e:
        # Clean up any partially created files
        if video_path.exists():
            os.unlink(video_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error saving translation: {str(e)}"
        )


@app.get("/user/{user_id}/translations/with-text")
async def get_translations_with_text(user_id: int, request: Request):
    """
    Get all saved translations with text for a user

    Args:
        user_id: ID of the user

    Returns:
        List of saved translations with text
    """
    user_dir = STORAGE_ROOT / "users" / str(user_id)
    saved_dir = user_dir / "saved"

    # Check if user directory exists
    if not user_dir.exists():
        return {"translations": []}

    # Check if the saved directory exists
    if not saved_dir.exists():
        return {"translations": []}

    try:
        # Get all text files in the translations directory
        translations_dir = saved_dir / "translations"
        if not translations_dir.exists():
            return {"translations": []}

        # List all text files
        text_files = list(translations_dir.glob("*.txt"))

        translations = []
        for text_file in text_files:
            translation_id = text_file.stem  # Get filename without extension

            # Read the text content
            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    text = f.read()
            except:
                text = "Error reading translation"


            # Get metadata if available
            metadata_file = saved_dir / "metadata" / f"{translation_id}.json"
            timestamp = ""
            filename = "unknown"

            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        timestamp = metadata.get("timestamp", "")
                        filename = metadata.get("original_filename", "unknown")
                except:
                    pass

            # Check if video exists
            video_file = saved_dir / "videos" / f"{translation_id}.mp4"
            video_url = None
            if video_file.exists():
                # Use full URL with domain
                video_url = f"{request.base_url}user/{user_id}/translations/video/{translation_id}"

            thumbnail_file = saved_dir / "thumbnails" / f"{translation_id}.jpg"
            thumbnail_url = None
            if thumbnail_file.exists():
              thumbnail_url = f"{request.base_url}user/{user_id}/translations/thumbnail/{translation_id}"
            # Add to results
            translations.append({
                "id": translation_id,
                "text": text,
                "timestamp": timestamp,
                "filename": filename,
                "video_url": video_url,
                "thumbnail_url": thumbnail_url
            })

        # Sort by ID (which should be sequential) for consistency
        translations.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else 0, reverse=True)

        return {"translations": translations}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving translations: {str(e)}"
        )

@app.get("/user/{user_id}/translations/with-audio")
async def get_translations_with_audio(user_id: int, request: Request):
    """
    Get all saved translations with audio for a user

    Args:
        user_id: ID of the user
        request: Request object to get base URL

    Returns:
        List of saved translations with audio
    """
    user_dir = STORAGE_ROOT / "users" / str(user_id)
    saved_dir = user_dir / "saved"

    # Check if user directory exists
    if not user_dir.exists():
        return {"translations": []}

    # Check if the saved directory exists
    if not saved_dir.exists():
        return {"translations": []}

    try:
        # Get all audio files in the audio directory
        audio_dir = saved_dir / "audio"
        if not audio_dir.exists():
            return {"translations": []}

        # List all mp3 files
        audio_files = list(audio_dir.glob("*.mp3"))

        translations = []
        for audio_file in audio_files:
            translation_id = audio_file.stem  # Get filename without extension

            # Read the text content if available
            text_file = saved_dir / "translations" / f"{translation_id}.txt"
            text = ""

            if text_file.exists():
                try:
                    with open(text_file, "r", encoding="utf-8") as f:
                        text = f.read()
                except:
                    text = "Error reading translation"

            # Get metadata if available
            metadata_file = saved_dir / "metadata" / f"{translation_id}.json"
            timestamp = ""
            filename = "unknown"

            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        timestamp = metadata.get("timestamp", "")
                        filename = metadata.get("original_filename", "unknown")
                except:
                    pass

            # Check if video exists
            video_file = saved_dir / "videos" / f"{translation_id}.mp4"
            video_url = None
            if video_file.exists():
                # Use full URL with domain
                video_url = f"{request.base_url}user/{user_id}/translations/video/{translation_id}"

            thumbnail_file = saved_dir / "thumbnails" / f"{translation_id}.jpg"
            thumbnail_url = None
            if thumbnail_file.exists():
              thumbnail_url = f"{request.base_url}user/{user_id}/translations/thumbnail/{translation_id}"


            # Add to results with full URLs
            translations.append({
                "id": translation_id,
                "text": text,
                "timestamp": timestamp,
                "filename": filename,
                "audio_url": f"{request.base_url}user/{user_id}/translations/audio/{translation_id}",
                "video_url": video_url,
                "thumbnail_url": thumbnail_url
            })

        # Sort by ID (which should be sequential) for consistency
        translations.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else 0, reverse=True)

        return {"translations": translations}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving translations with audio: {str(e)}"
        )


@app.get("/user/{user_id}/translations/audio/{translation_id}")
async def get_translation_audio(user_id: int, translation_id: str):
    """
    Get audio for a specific translation

    Args:
        user_id: ID of the user
        translation_id: ID of the translation

    Returns:
        Audio file
    """
    audio_path = STORAGE_ROOT / "users" / str(user_id) / "saved" / "audio" / f"{translation_id}.mp3"

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    # Open the file in binary mode and read the content
    try:
        with open(audio_path, "rb") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading audio file: {str(e)}")

    # Return the audio file with proper headers
    return Response(
        content=content,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"inline; filename={translation_id}.mp3"}
    )

@app.get("/user/{user_id}/translations/video/{translation_id}")
async def get_translation_video(user_id: int, translation_id: str):
    """
    Get video for a specific translation

    Args:
        user_id: ID of the user
        translation_id: ID of the translation

    Returns:
        Video file
    """
    video_path = STORAGE_ROOT / "users" / str(user_id) / "saved" / "videos" / f"{translation_id}.mp4"

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Get original filename if available
    original_filename = f"{translation_id}.mp4"
    metadata_path = STORAGE_ROOT / "users" / str(user_id) / "saved" / "metadata" / f"{translation_id}.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if "original_filename" in metadata:
                    original_filename = metadata["original_filename"]
        except:
            pass

    # Return the video file
    return Response(
        content=open(video_path, "rb").read(),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{original_filename}"'}
    )


@app.get("/user/{user_id}/v2t/recent-translations")
async def get_recent_v2t_translations(user_id: int):
    """
    Get metadata for recent translations with thumbnails and audio URLs

    Args:
        user_id: ID of the user

    Returns:
        List of metadata with text, thumbnail URLs and audio URLs
    """
    user_dir = STORAGE_ROOT / "users" / str(user_id)
    recent_dir = user_dir / "recent_translations"
    metadata_dir = recent_dir / "metadata"
    translations_dir = recent_dir / "translations"

    # Check if directories exist
    if not user_dir.exists() or not metadata_dir.exists():
        return {"translations": []}

    try:
        # Get all JSON files in metadata directory
        metadata_files = list(metadata_dir.glob("*.json"))

        translations = []
        for metadata_file in metadata_files:
            try:
                # Read the metadata JSON file
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                translation_id = metadata.get("translation_id")
                if not translation_id:
                    continue

                # Add text content if available
                text_file = translations_dir / f"{translation_id}.txt"
                if text_file.exists():
                    try:
                        with open(text_file, "r", encoding="utf-8") as f:
                            metadata["text"] = f.read()
                    except Exception as e:
                        print(f"Error reading text file {text_file}: {str(e)}")

                # URLs are already in the metadata
                translations.append(metadata)

            except Exception as e:
                # Skip files with errors
                print(f"Error reading metadata file {metadata_file}: {str(e)}")
                continue

        # Sort by timestamp (newest first)
        translations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return {"translations": translations}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving recent translations: {str(e)}"
        )


@app.get("/user/{user_id}/recent-translations/audio/{translation_id}")
async def get_recent_translation_audio(user_id: int, translation_id: str):
    """
    Get audio for a recent translation

    Args:
        user_id: ID of the user
        translation_id: ID of the translation

    Returns:
        Audio file
    """
    audio_path = STORAGE_ROOT / "users" / str(user_id) / "recent_translations" / "audio" / f"{translation_id}.mp3"

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    # Open the file in binary mode and read the content
    try:
        with open(audio_path, "rb") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading audio file: {str(e)}")

    # Return the audio file with proper headers
    return Response(
        content=content,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"inline; filename={translation_id}.mp3"}
    )



@app.get("/user/{user_id}/v2t/recent-saves")
async def get_recent_v2t_saves(user_id: int, request: Request):
    """
    Get the 3 most recent saved translations for a user

    Args:
        user_id: ID of the user
        request: Request object to get base URL

    Returns:
        List of the 3 most recent saved translations with URLs
    """
    user_dir = STORAGE_ROOT / "users" / str(user_id)
    recent_saves_path = user_dir / "recent_saves.json"
    saved_dir = user_dir / "saved"

    # Check if user directory exists
    if not user_dir.exists():
        return {"saves": []}

    # Check if recent_saves.json exists
    if not recent_saves_path.exists():
        return {"saves": []}

    try:
        # Read the recent_saves.json file
        with open(recent_saves_path, "r") as f:
            recent_saves = json.load(f)

        # Enhance saves with URLs and additional info
        enhanced_saves = []

        for save in recent_saves:
            save_id = save.get("id")
            if not save_id:
                continue

            # Check for metadata to determine if audio is available
            metadata_path = saved_dir / "metadata" / f"{save_id}.json"
            has_audio = False

            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        has_audio = metadata.get("has_audio", False)
                except:
                    pass

            # Create a copy of the save to enhance
            enhanced_save = dict(save)

            thumbnail_path = saved_dir / "thumbnails" / f"{save_id}.jpg"
            if thumbnail_path.exists():
              enhanced_save["thumbnail_url"] = f"{request.base_url}user/{user_id}/translations/thumbnail/{save_id}"

            # Add has_audio flag
            enhanced_save["has_audio"] = has_audio

            # Add URLs for video and audio if they exist
            video_path = saved_dir / "videos" / f"{save_id}.mp4"
            if video_path.exists():
                enhanced_save["video_url"] = f"{request.base_url}user/{user_id}/translations/video/{save_id}"

            if has_audio:
                audio_path = saved_dir / "audio" / f"{save_id}.mp3"
                if audio_path.exists():
                    enhanced_save["audio_url"] = f"{request.base_url}user/{user_id}/translations/audio/{save_id}"

                    # Get text from audio directory
                    text_path = saved_dir / "audio" / f"{save_id}.txt"
                else:
                    enhanced_save["has_audio"] = False
            else:
                # Get text from translations directory
                text_path = saved_dir / "translations" / f"{save_id}.txt"

            # Add text if it exists
            if 'text_path' in locals() and text_path.exists():
                try:
                    with open(text_path, "r", encoding="utf-8") as f:
                        enhanced_save["text"] = f.read()
                except:
                    enhanced_save["text"] = "Error reading translation"

            enhanced_saves.append(enhanced_save)

        return {"saves": enhanced_saves}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving recent saves: {str(e)}"
        )


class DeleteRequest(BaseModel):
    translation_id: str
    user_id: Union[int, str]

@app.delete("/user/{user_id}/translations/with-text/{translation_id}")
async def delete_text_translation(user_id: int, translation_id: str):
    """
    Delete a saved translation with text

    Args:
        user_id: ID of the user
        translation_id: ID of the translation to delete

    Returns:
        Success message
    """
    try:
        user_id = int(user_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="User ID must be convertible to an integer"
        )

    user_dir = STORAGE_ROOT / "users" / str(user_id)
    saved_dir = user_dir / "saved"

    # Define paths to files
    video_path = saved_dir / "videos" / f"{translation_id}.mp4"
    text_path = saved_dir / "translations" / f"{translation_id}.txt"
    metadata_path = saved_dir / "metadata" / f"{translation_id}.json"
    thumbnail_path = saved_dir / "thumbnails" / f"{translation_id}.jpg"

    # Check if the text file exists (primary indicator for text-based translation)
    if not text_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Text translation {translation_id} not found"
        )

    try:
        # Delete files if they exist
        if text_path.exists():
            os.unlink(text_path)
        if video_path.exists():
            os.unlink(video_path)
        if metadata_path.exists():
            os.unlink(metadata_path)
        if thumbnail_path.exists():
            os.unlink(thumbnail_path)

        # Update recent_saves.json to remove this translation
        recent_saves_path = user_dir / "recent_saves.json"
        if recent_saves_path.exists():
            try:
                with open(recent_saves_path, "r") as f:
                    recent_saves = json.load(f)

                # Filter out the deleted translation
                recent_saves = [save for save in recent_saves if save.get("id") != translation_id]

                # Write the updated list back
                with open(recent_saves_path, "w") as f:
                    json.dump(recent_saves, f)
            except Exception as e:
                print(f"Error updating recent_saves.json: {str(e)}")

        return {"success": True, "message": f"Text translation {translation_id} deleted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting text translation: {str(e)}"
        )

@app.delete("/user/{user_id}/translations/with-audio/{translation_id}")
async def delete_audio_translation(user_id: int, translation_id: str):
    """
    Delete a saved translation with audio

    Args:
        user_id: ID of the user
        translation_id: ID of the translation to delete

    Returns:
        Success message
    """
    try:
        user_id = int(user_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="User ID must be convertible to an integer"
        )

    user_dir = STORAGE_ROOT / "users" / str(user_id)
    saved_dir = user_dir / "saved"

    # Define paths to files
    video_path = saved_dir / "videos" / f"{translation_id}.mp4"
    audio_path = saved_dir / "audio" / f"{translation_id}.mp3"
    audio_text_path = saved_dir / "audio" / f"{translation_id}.txt"
    metadata_path = saved_dir / "metadata" / f"{translation_id}.json"
    thumbnail_path = saved_dir / "thumbnails" / f"{translation_id}.jpg"

    # Check if the audio file exists (primary indicator for audio-based translation)
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Audio translation {translation_id} not found"
        )

    try:
        # Delete files if they exist
        if audio_path.exists():
            os.unlink(audio_path)
        if audio_text_path.exists():
            os.unlink(audio_text_path)
        if video_path.exists():
            os.unlink(video_path)
        if metadata_path.exists():
            os.unlink(metadata_path)
        if thumbnail_path.exists():
            os.unlink(thumbnail_path)

        # Update recent_saves.json to remove this translation
        recent_saves_path = user_dir / "recent_saves.json"
        if recent_saves_path.exists():
            try:
                with open(recent_saves_path, "r") as f:
                    recent_saves = json.load(f)

                # Filter out the deleted translation
                recent_saves = [save for save in recent_saves if save.get("id") != translation_id]

                # Write the updated list back
                with open(recent_saves_path, "w") as f:
                    json.dump(recent_saves, f)
            except Exception as e:
                print(f"Error updating recent_saves.json: {str(e)}")

        return {"success": True, "message": f"Audio translation {translation_id} deleted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting audio translation: {str(e)}"
        )


@app.get("/user/{user_id}/search/v2t")
async def search_v2t_translations(user_id: int, query: str, request: Request):
    """
    Search for translations containing the query text

    Args:
        user_id: ID of the user
        query: Text to search for
        request: Request object to get base URL

    Returns:
        List of matching translations
    """
    if not query or query.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty"
        )

    try:
        user_id = int(user_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="User ID must be an integer"
        )

    user_dir = STORAGE_ROOT / "users" / str(user_id)
    saved_dir = user_dir / "saved"

    # Check if user directory exists
    if not user_dir.exists() or not saved_dir.exists():
        return {"results": [], "total": 0, "query": query}

    # Define locations to search
    text_translations_dir = saved_dir / "translations"
    audio_text_dir = saved_dir / "audio"

    # Initialize results
    results = []
    matched_ids = set()  # To track IDs we've already matched

    try:
        # Case-insensitive search
        query = query.lower()

        # Helper function to process a match
        def process_match(file_path, is_audio):
            translation_id = file_path.stem

            # Skip if we've already added this ID
            if translation_id in matched_ids:
                return

            # Read the text content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Check if query is in the text (case insensitive)
                if query not in text.lower():
                    return

                # Get metadata
                metadata_file = saved_dir / "metadata" / f"{translation_id}.json"
                timestamp = ""
                filename = "unknown"
                has_audio = is_audio

                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            timestamp = metadata.get("timestamp", "")
                            filename = metadata.get("original_filename", "unknown")
                            has_audio = metadata.get("has_audio", is_audio)
                    except:
                        pass

                # Create result object
                result = {
                    "id": translation_id,
                    "text": text,
                    "timestamp": timestamp,
                    "filename": filename,
                    "type": "audio" if is_audio else "text",
                    "has_audio": has_audio
                }

                # Create context with highlighted match
                text_lower = text.lower()
                start_pos = text_lower.find(query)
                if start_pos >= 0:
                    # Get some context around the match
                    context_start = max(0, start_pos - 30)
                    context_end = min(len(text), start_pos + len(query) + 30)

                    # Extract the context
                    if context_start > 0:
                        prefix = "..." + text[context_start:start_pos]
                    else:
                        prefix = text[:start_pos]

                    if context_end < len(text):
                        suffix = text[start_pos + len(query):context_end] + "..."
                    else:
                        suffix = text[start_pos + len(query):]

                    # The matched text
                    matched = text[start_pos:start_pos + len(query)]

                    # Create the highlighted context
                    result["match_context"] = f"{prefix}<highlight>{matched}</highlight>{suffix}"

                # Add URLs for video and audio
                video_path = saved_dir / "videos" / f"{translation_id}.mp4"
                if video_path.exists():
                    result["video_url"] = f"{request.base_url}user/{user_id}/translations/video/{translation_id}"

                if has_audio:
                    audio_path = saved_dir / "audio" / f"{translation_id}.mp3"
                    if audio_path.exists():
                        result["audio_url"] = f"{request.base_url}user/{user_id}/translations/audio/{translation_id}"

                # Add to results and track the ID
                results.append(result)
                matched_ids.add(translation_id)

            except Exception as e:
                print(f"Error processing match for {file_path}: {str(e)}")

        # Search in text translations
        if text_translations_dir.exists():
            for text_file in text_translations_dir.glob("*.txt"):
                process_match(text_file, False)

        # Search in audio text files
        if audio_text_dir.exists():
            for text_file in audio_text_dir.glob("*.txt"):
                process_match(text_file, True)

        # Sort by timestamp (newest first) or by ID if timestamps are equal
        results.sort(key=lambda x: (x.get("timestamp", ""), x.get("id", "")), reverse=True)

        return {
            "results": results,
            "total": len(results),
            "query": query
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching translations: {str(e)}"
        )


@app.get("/user/{user_id}/translations/thumbnail/{translation_id}")
async def get_translation_thumbnail(user_id: int, translation_id: str):
    """
    Get thumbnail for a specific translation

    Args:
        user_id: ID of the user
        translation_id: ID of the translation

    Returns:
        Thumbnail image
    """
    thumbnail_path = STORAGE_ROOT / "users" / str(user_id) / "saved" / "thumbnails" / f"{translation_id}.jpg"

    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    # Open the file in binary mode and read the content
    try:
        with open(thumbnail_path, "rb") as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading thumbnail file: {str(e)}")

    # Return the thumbnail with proper headers
    return Response(
        content=content,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename={translation_id}.jpg"}
    )



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger("handtalk-api")

# Translation cache to store recent translations
TRANSLATION_CACHE = OrderedDict()
MAX_CACHE_ITEMS = 20  # Store only 20 recent translations
CACHE_EXPIRY_SECONDS = 1200

def extract_frame_from_gif(gif_data, output_path, frame_number=None):
    """
    Extract a specific frame from a GIF and save it as a JPEG at full size and quality

    Args:
        gif_data: The binary GIF data
        output_path: Path to save the frame as JPEG
        frame_number: The index of the frame to extract (if None, uses middle frame)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a BytesIO object from the binary data
        with io.BytesIO(gif_data) as gif_io:
            # Open the GIF
            with Image.open(gif_io) as gif:
                # Count total frames
                frame_count = 0
                try:
                    while True:
                        gif.seek(frame_count)
                        frame_count += 1
                except EOFError:
                    # End of frames
                    pass

                # If frame_number is None, use the middle frame
                if frame_number is None:
                    frame_to_extract = frame_count // 2  # Integer division to get the middle frame
                    logger.info(f"Using middle frame: {frame_to_extract} of {frame_count} total frames")
                elif frame_number >= frame_count:
                    logger.warning(f"Requested frame {frame_number} but GIF only has {frame_count} frames. Using last frame.")
                    frame_to_extract = frame_count - 1
                else:
                    frame_to_extract = frame_number

                # Seek to the desired frame
                gif.seek(frame_to_extract)

                # Convert to RGB if needed (some GIFs might have transparency)
                if gif.mode != 'RGB':
                    extracted_frame = gif.convert('RGB')
                else:
                    extracted_frame = gif.copy()

                # Save as JPEG with maximum quality
                extracted_frame.save(output_path, format="JPEG", quality=100)

                logger.info(f"Saved frame {frame_to_extract} to {output_path}")
                return True

    except Exception as e:
        logger.error(f"Error extracting frame from GIF: {e}")
        return False


def update_recent_t2v_translations(user_id: int, translation_data: dict, img_base64: str, request: Request = None):
    """
    Update the list of recent translations for a user

    Args:
        user_id: ID of the user
        translation_data: Dictionary containing translation data
        img_base64: Base64 encoded image data for the translation
        request: Request object to generate full URLs
    """
    try:
        # Get user directory and create recent_translations directory if needed
        user_dir, _, thumbnails_dir, _, _ = ensure_user_directory(user_id)
        recent_dir = os.path.join(user_dir, "recent_translations")
        recent_thumbnails_dir = os.path.join(recent_dir, "thumbnails")
        os.makedirs(recent_dir, exist_ok=True)
        os.makedirs(recent_thumbnails_dir, exist_ok=True)

        # Path to the JSON file storing recent translations
        recent_file = os.path.join(recent_dir, "recent.json")

        # Read existing recent translations or create empty list
        recent_translations = []
        if os.path.exists(recent_file):
            try:
                with open(recent_file, 'r') as f:
                    recent_translations = json.load(f)
            except json.JSONDecodeError:
                recent_translations = []

        # Create a unique ID for this translation
        translation_id = int(time.time())

        # Generate and save thumbnail
        has_thumbnail = False
        thumbnail_url = None

        try:
            # Decode base64 to binary
            gif_binary = base64.b64decode(img_base64)

            # Save thumbnail
            thumbnail_path = os.path.join(recent_thumbnails_dir, f"{translation_id}.jpg")
            has_thumbnail = extract_frame_from_gif(gif_binary, thumbnail_path)

            # Generate thumbnail URL
            if has_thumbnail and request:
                thumbnail_url = f"{request.base_url}user/{user_id}/recent-translations/t2v/thumbnail/{translation_id}"
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for recent translation: {e}")

        # Create the new entry
        new_entry = {
            "id": translation_id,
            "timestamp": datetime.now().isoformat(),
            "text": translation_data.get("original_text", ""),
            "words": translation_data.get("words", []),
            "has_thumbnail": has_thumbnail,
            "thumbnail_url": thumbnail_url
        }

        # Add to the beginning of the list
        recent_translations.insert(0, new_entry)

        # Keep only the 3 most recent
        recent_translations = recent_translations[:3]

        # Delete thumbnails for translations that were removed from the list
        if len(recent_translations) < 3:
            # Find thumbnails that don't correspond to current translations
            current_ids = [entry["id"] for entry in recent_translations]
            for filename in os.listdir(recent_thumbnails_dir):
                if filename.endswith(".jpg"):
                    try:
                        file_id = int(filename.split(".")[0])
                        if file_id not in current_ids:
                            os.remove(os.path.join(recent_thumbnails_dir, filename))
                            logger.info(f"Deleted old recent translation thumbnail: {filename}")
                    except ValueError:
                        pass

        # Save the updated list
        with open(recent_file, 'w') as f:
            json.dump(recent_translations, f, indent=2)

        logger.info(f"Updated recent translations for user {user_id}")

    except Exception as e:
        logger.error(f"Error updating recent translations: {e}")


def generate_audio_from_text(text: str, output_path: str) -> bool:
    """
    Generate audio from text using gTTS

    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file

    Returns:
        True if successful, False otherwise
    """
    try:
        tts = gtts.gTTS(text)
        tts.save(output_path)
        logger.info(f"Generated audio saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return False

# ======================
# DATA MODELS
# ======================

class SignLanguageResponse(BaseModel):
    img: str
    words: List[str]
    processing_time: float
    word_count: int
    original_text: str
    isSaved: bool = False
    translation_id: Optional[int] = None

class TranslationItem(BaseModel):
    translation_id: int
    original_text: str
    words: List[str]
    img: str
    saved_at: str
    thumbnail_url: Optional[str] = None
    audio_url: Optional[str] = None
    has_audio: bool = False
    has_thumbnail: bool = False

class UserTranslationsResponse(BaseModel):
    translations: List[TranslationItem]
    count: int

class SimpleSaveRequest(BaseModel):
    user_id: int
    text: str
    generate_audio: bool = False

class DeleteTranslationRequest(BaseModel):
    user_id: int

# ======================
# CUSTOM EXCEPTIONS
# ======================

class GlossNotFoundException(Exception):
    def __init__(self, gloss: str):
        self.gloss = gloss
        super().__init__(f"No gloss found for: {gloss}")

class PoseNotFoundException(Exception):
    def __init__(self, words: List[str]):
        self.words = words
        super().__init__(f"No pose found for words: {', '.join(words)}")

class CacheNotFoundException(Exception):
    def __init__(self, user_id: int, text: str):
        self.user_id = user_id
        self.text = text
        super().__init__(f"Translation not found in cache for: {text}")



# ======================
# CORE UTILITIES
# ======================

def add_to_cache(user_id: int, text, response_data):
    """Add a translation to the cache with timestamp."""
    # Remove oldest item if cache is full
    if len(TRANSLATION_CACHE) >= MAX_CACHE_ITEMS:
        TRANSLATION_CACHE.popitem(last=False)

    # Add new item
    TRANSLATION_CACHE[(user_id, text)] = {
        "data": response_data,
        "timestamp": time.time()
    }
    logger.info(f"Added to cache: {user_id}, {text}")


def get_from_cache(user_id: int, text):
    """Get a translation from cache if it exists and is not expired."""
    cache_key = (user_id, text)
    if cache_key not in TRANSLATION_CACHE:
        return None

    cache_entry = TRANSLATION_CACHE[cache_key]

    # Check if cache entry is expired
    if time.time() - cache_entry["timestamp"] > CACHE_EXPIRY_SECONDS:
        del TRANSLATION_CACHE[cache_key]
        logger.info(f"Cache entry expired: {user_id}, {text}")
        return None

    # Move this item to the end (most recently used)
    TRANSLATION_CACHE.move_to_end(cache_key)
    logger.info(f"Cache hit: {user_id}, {text}")

    return cache_entry["data"]


# Add these helper functions
def ensure_user_directory(user_id: int) -> tuple:
    """Create directory structure for a user and return paths."""
    user_dir = os.path.join("translations", str(user_id))
    base64_dir = os.path.join(user_dir, "base64_data")
    thumbnails_dir = os.path.join(user_dir, "thumbnails")
    audio_dir = os.path.join(user_dir, "audio")
    metadata_path = os.path.join(user_dir, "metadata.json")

    # Create directories if they don't exist
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(base64_dir, exist_ok=True)
    os.makedirs(thumbnails_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    return user_dir, base64_dir, thumbnails_dir, audio_dir, metadata_path


def get_user_metadata(metadata_path: str) -> dict:
    """Get existing metadata or create new metadata structure."""
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                # If file exists but is invalid JSON, create new metadata
                return {"translations": []}
    return {"translations": []}


def check_if_translation_saved(user_id: int, text: str) -> Tuple[bool, Optional[int]]:
    """Check if a translation is already saved for this user.

    Returns:
        tuple: (is_saved, translation_id if saved else None)
    """
    # Setup directory structure
    _, _, _, _, metadata_path = ensure_user_directory(user_id)

    # Get existing metadata
    metadata = get_user_metadata(metadata_path)

    # Normalize input text for comparison
    text = text.strip().lower()

    # Check if the text is already saved
    for translation in metadata["translations"]:
        if translation["original_text"].strip().lower() == text:
            return True, translation["translation_id"]

    return False, None

# ======================
# FASTAPI SETUP
# ======================

# Load the pose lookup data
try:
    POSE_LOOKUP = glosstopose.PoseLookup("gloss2pose", "asl")
    warnings.filterwarnings("ignore")
except Exception as e:
    logger.critical(f"Failed to initialize POSE_LOOKUP: {e}")
    raise


# ======================
# EXCEPTION HANDLERS
# ======================

@app.exception_handler(GlossNotFoundException)
async def gloss_not_found_handler(request: Request, exc: GlossNotFoundException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Gloss not found", "detail": str(exc)}
    )

@app.exception_handler(PoseNotFoundException)
async def pose_not_found_handler(request: Request, exc: PoseNotFoundException):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Pose not found", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.exception_handler(CacheNotFoundException)
async def cache_not_found_handler(request: Request, exc: CacheNotFoundException):
    logger.warning(f"Cache miss: Translation not found for '{exc.text}'")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Cache miss", "detail": "Translation not found in cache. Please translate this text first."}
    )


# ======================
# MIDDLEWARE
# ======================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log request details
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Request: {request.method} {request.url.path} from {client_host}")

    # Process the request
    response = await call_next(request)

    # Log response details
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.4f}s")

    return response


# ======================
# API ENDPOINTS
# ======================

@app.get(
    "/gloss2pose",
    response_model=SignLanguageResponse,
    summary="Convert text to sign language",
    description="Converts text input to animated sign language GIF"
)
async def make_pose(
    gloss: str = Query(None, description="Text to convert to sign language"),
    user_id: int = Query(None, description="User ID"),
    request: Request = None
):

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID is required"
        )

    if not gloss:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No gloss provided"
        )

    # Normalize input
    gloss = gloss.strip().lower()
    try:
        # 1. Check cache first
        cached_response = get_from_cache(user_id, gloss)
        if cached_response:
            # Check if this text is saved, update cached response accordingly
            is_saved, translation_id = check_if_translation_saved(user_id, gloss)
            if is_saved:
                cached_response["isSaved"] = True
                cached_response["translation_id"] = translation_id
            else:
                cached_response["isSaved"] = False
                cached_response["translation_id"] = None

            update_recent_t2v_translations(user_id, cached_response, cached_response["img"], request)
            return cached_response

        # 2. Cache miss: check saved translation
        is_saved, translation_id = check_if_translation_saved(user_id, gloss)
        # If it's saved, check if we can retrieve it directly
        if is_saved:
            # Try to get data from the saved translation
            user_dir, base64_dir,  _, audio_dir, metadata_path = ensure_user_directory(user_id)
            metadata = get_user_metadata(metadata_path)
            saved_translation = next((t for t in metadata["translations"] if t["translation_id"] == translation_id), None)

            if saved_translation:
                file_path = os.path.join(base64_dir, saved_translation["filename"])
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        translation_data = json.load(f)

                    response_data = {
                        "img": translation_data["img"],
                        "words": saved_translation["words"],
                        "processing_time": 0.0,
                        "word_count": len(saved_translation["words"]),
                        "original_text": saved_translation["original_text"],
                        "isSaved": True,
                        "translation_id": translation_id
                    }

                    add_to_cache(user_id, gloss, response_data)
                    update_recent_t2v_translations(user_id, response_data, response_data["img"], request)
                    return response_data

        # 3. Not cached or saved: process normally
        start = time.time()
        logger.info(f"Processing gloss: {gloss}")

        # Prepare glosses
        glosses = glosstopose.prepare_glosses(gloss)
        if not glosses:
            raise GlossNotFoundException(gloss)

        # Get pose data
        logger.info(f"Getting pose data for {len(glosses)} glosses")
        pose_start = time.time()
        pose, words = POSE_LOOKUP.gloss_to_pose(glosses)
        pose_time = time.time() - pose_start
        logger.info(f"Pose generation took {pose_time:.4f}s")

        if not pose:
            raise PoseNotFoundException(glosses)

        logger.info("Generating visualization")
        viz_start = time.time()
        glosstopose.scale_down(pose, 512)

        p = PoseVisualizer(pose, thickness=2)

        # Use CUDA if available for the visualization
        if device.type == 'cuda':
            try:
                # Use mixed precision for faster processing
                with torch.cuda.amp.autocast():
                    img = p.save_gif(None, p.draw())
                logger.info("Visualization used GPU acceleration")
            except Exception as e:
                logger.warning(f"Failed to use GPU for visualization: {e}, falling back to CPU")
                img = p.save_gif(None, p.draw())
        else:
            img = p.save_gif(None, p.draw())

        viz_time = time.time() - viz_start
        logger.info(f"Visualization took {viz_time:.4f}s")
        img_base64 = base64.b64encode(img).decode('utf-8')
        processing_time = time.time() - start
        logger.info(f"Total processing time: {processing_time:.4f}s")

        # Return JSON with encoded image and metadata
        response_data = {
            "img": img_base64,
            "words": words,
            "processing_time": processing_time,
            "word_count": len(words),
            "original_text": gloss,
            "isSaved": False,
            "translation_id":None
        }

        add_to_cache(user_id, gloss, response_data)

        update_recent_t2v_translations(user_id, response_data, img_base64, request)

        # Return the response
        return response_data


    except (GlossNotFoundException, PoseNotFoundException) as e:
        # Let the specific exception handlers deal with these
        raise
    except Exception as e:
        logger.error(f"Error processing gloss: {gloss}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/t2v/save", summary="Save a text-to-sign translation")
async def save_t2v_translation(data: SimpleSaveRequest, request: Request = None):
    try:
        logger.info(f"Save request received for text: '{data.text}'")

        if data.user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID is required"
            )

        # Get the translation data from cache
        cached_data = get_from_cache(data.user_id, data.text)
        if not cached_data:
            logger.warning(f"Cache miss during save: user_id={data.user_id}, text={data.text}")
            raise CacheNotFoundException(data.user_id, data.text)

        # Setup directory structure
        user_dir, base64_dir, thumbnails_dir, audio_dir, metadata_path = ensure_user_directory(data.user_id)

        # Get existing metadata
        metadata = get_user_metadata(metadata_path)

        # Generate new translation ID
        new_id = 1
        if metadata["translations"]:
            new_id = max(t["translation_id"] for t in metadata["translations"]) + 1

        # Generate timestamp
        timestamp = time.time()

        # Save the base64 data to its own file
        translation_filename = f"translation_{new_id}.json"
        base64_path = os.path.join(base64_dir, translation_filename)

        with open(base64_path, "w") as f:
            json.dump({"img": cached_data["img"]}, f)

        logger.info(f"Saved base64 data to file: {translation_filename}")

        has_thumbnail = False
        has_audio = False
        audio_url = None

        try:
            gif_binary = base64.b64decode(cached_data["img"])
            thumbnail_path = os.path.join(thumbnails_dir, f"{new_id}.jpg")
            has_thumbnail = extract_frame_from_gif(gif_binary, thumbnail_path)
            logger.info(f"Thumbnail generated: {has_thumbnail}")
        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            has_thumbnail = False

        if data.generate_audio:
            try:
                audio_path = os.path.join(audio_dir, f"{new_id}.mp3")
                has_audio = generate_audio_from_text(data.text, audio_path)

                if has_audio and request:
                    audio_url = f"{request.base_url}user/{data.user_id}/t2v/audio/{new_id}"
                    logger.info(f"Generated audio for translation #{new_id}")

            except Exception as e:
                logger.error(f"Failed to generate audio: {e}")
                has_audio = False


        # Add metadata entry
        translation_metadata = {
            "translation_id": new_id,
            "original_text": cached_data["original_text"],
            "words": cached_data["words"],
            "filename": translation_filename,
            "has_thumbnail": has_thumbnail,
            "has_audio": has_audio,
            "saved_at": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        }

        metadata["translations"].append(translation_metadata)

        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


        logger.info(f"Saved translation #{new_id} for text: '{data.text}'")

        return {
            "translation_id": new_id,
            "message": "Translation saved successfully",
            "has_thumbnail": has_thumbnail,
            "audio_url": audio_url,
            "saved_at": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        }

    except CacheNotFoundException:
        raise

    except Exception as e:
        logger.error(f"Error saving translation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save translation: {str(e)}"
        )


@app.delete(
    "/delete_translations/{translation_id}",
    summary="Delete a saved translation",
    description="Deletes a specific translation by its translation ID and the given user ID"
)
async def delete_translation(
    translation_id: int,
    request: DeleteTranslationRequest = Query(...)
):
    try:
        logger.info(f"Delete request for translation ID: {translation_id}")

        if request.user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID is required"
            )

        user_dir, base64_dir, thumbnails_dir, audio_dir, metadata_path = ensure_user_directory(request.user_id)
        metadata = get_user_metadata(metadata_path)

        # Find the translation to delete
        translation_to_delete = None
        remaining_translations = []
        for translation in metadata["translations"]:
            if translation["translation_id"] == translation_id:
                translation_to_delete = translation
            else:
                remaining_translations.append(translation)

        if not translation_to_delete:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Translation with ID {translation_id} not found"
            )

        # Delete the file
        file_path = os.path.join(base64_dir, translation_to_delete["filename"])
        thumbnail_path = os.path.join(thumbnails_dir, f"{translation_id}.jpg")
        audio_path = os.path.join(audio_dir, f"{translation_id}.mp3")

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")

            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                logger.info(f"Deleted thumbnail: {thumbnail_path}")

            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Deleted audio: {audio_path}")


        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete translation file"
            )

        # Reorganize IDs and filenames for remaining translations
        new_translations = []
        temp_dir = os.path.join(user_dir, "temp")
        temp_thumbnails_dir = os.path.join(user_dir, "temp_thumbnails")
        temp_audio_dir = os.path.join(user_dir, "temp_audio")

        os.makedirs(temp_thumbnails_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(temp_audio_dir, exist_ok=True)

        try:
            # First move all remaining files to temp dir with new names
            for idx, translation in enumerate(remaining_translations, 1):
                old_path = os.path.join(base64_dir, translation["filename"])
                new_filename = f"translation_{idx}.json"
                new_path = os.path.join(temp_dir, new_filename)

                if os.path.exists(old_path):
                    os.rename(old_path, new_path)

                old_thumbnail = os.path.join(thumbnails_dir, f"{translation['translation_id']}.jpg")
                new_thumbnail = os.path.join(temp_thumbnails_dir, f"{idx}.jpg")
                has_thumbnail = False

                if os.path.exists(old_thumbnail):
                    os.rename(old_thumbnail, new_thumbnail)
                    has_thumbnail = True
                    logger.info(f"Renamed thumbnail {translation['translation_id']} to {idx}")

                old_audio = os.path.join(audio_dir, f"{translation['translation_id']}.mp3")
                new_audio = os.path.join(temp_audio_dir, f"{idx}.mp3")
                has_audio = False
                if os.path.exists(old_audio):
                    os.rename(old_audio, new_audio)
                    has_audio = True
                    logger.info(f"Renamed audio {translation['translation_id']} to {idx}")

                # Update translation metadata
                new_translation = {
                    **translation,
                    "translation_id": idx,
                    "filename": new_filename,
                    "has_thumbnail": has_thumbnail,
                    "has_audio": has_audio
                }
                new_translations.append(new_translation)

            # Delete old base64_dir and rename temp dir
            import shutil

            if os.path.exists(base64_dir):
                shutil.rmtree(base64_dir)
            os.rename(temp_dir, base64_dir)


            if os.path.exists(thumbnails_dir):
                shutil.rmtree(thumbnails_dir)
            os.rename(temp_thumbnails_dir, thumbnails_dir)

            if os.path.exists(audio_dir):
                shutil.rmtree(audio_dir)
            os.rename(temp_audio_dir, audio_dir)

            # Update metadata
            metadata["translations"] = new_translations
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Reorganized IDs after deletion of {translation_id}")

            return {
                "success": True,
                "message": "Translation deleted",
                "deleted_translation_id": translation_id,
                "new_count": len(new_translations)
            }

        except Exception as e:
            logger.error(f"Error during reorganization: {e}")
            # Clean up temp dir if something went wrong
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(temp_thumbnails_dir):
                shutil.rmtree(temp_thumbnails_dir)
            if os.path.exists(temp_audio_dir):
                shutil.rmtree(temp_audio_dir)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reorganize translations after deletion"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting translation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete translation: {str(e)}"
        )

@app.get(
    "/retrieve_translations",
    response_model=UserTranslationsResponse,
    summary="Get all saved translations",
    description="Retrieves all saved translations including their animation data for a given user ID"
)
async def get_all_translations(
    user_id: int = Query(..., description="User ID"),
    request: Request = None  # Added request parameter to get base URL
):
    try:

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID is required"
            )

        # Get user directory structure
        user_dir, base64_dir, thumbnails_dir, audio_dir, metadata_path = ensure_user_directory(user_id)

        # Get metadata
        metadata = get_user_metadata(metadata_path)

        # Load each translation file
        translations = []
        for item in metadata["translations"]:
            try:
                file_path = os.path.join(base64_dir, item["filename"])
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        translation_data = json.load(f)

                    thumbnail_url = None
                    thumbnail_path = os.path.join(thumbnails_dir, f"{item['translation_id']}.jpg")
                    if os.path.exists(thumbnail_path) and request:
                        thumbnail_url = f"{request.base_url}user/{user_id}/t2v/thumbnail/{item['translation_id']}"

                    audio_url = None
                    audio_path = os.path.join(audio_dir, f"{item['translation_id']}.mp3")
                    if os.path.exists(audio_path) and request:
                        audio_url = f"{request.base_url}user/{user_id}/t2v/audio/{item['translation_id']}"

                    translations.append({
                        "translation_id": item["translation_id"],
                        "original_text": item["original_text"],
                        "words": item["words"],
                        "img": translation_data["img"],
                        "saved_at": item["saved_at"],
                        "thumbnail_url": thumbnail_url,
                        "audio_url": audio_url,
                        "has_audio": item.get("has_audio", False),
                        "has_thumbnail": item.get("has_thumbnail", False)
                    })
                else:
                    logger.warning(f"Translation file missing: {file_path}")
            except Exception as e:
                logger.error(f"Error loading translation {item['translation_id']}: {e}")


        if not translations:
            raise HTTPException(404, "No translations found")

        return {
            "translations": translations,
            "count": len(translations)
        }

    except Exception as e:
        logger.error(f"Error retrieving translations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve translations: {str(e)}"
        )


@app.get(
    "/user/{user_id}/t2v/thumbnail/{translation_id}",
    summary="Get thumbnail for a text-to-sign translation",
    description="Retrieves the thumbnail image for a specific text-to-sign translation"
)
async def get_t2v_thumbnail(user_id: int, translation_id: int):
    try:
        # Get user directory structure
        user_dir, _, thumbnails_dir, _, _ = ensure_user_directory(user_id)

        # Get thumbnail path
        thumbnail_path = os.path.join(thumbnails_dir, f"{translation_id}.jpg")

        if not os.path.exists(thumbnail_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thumbnail not found"
            )

        # Read the thumbnail file
        with open(thumbnail_path, "rb") as f:
            thumbnail_data = f.read()

        # Return the thumbnail with proper headers
        from fastapi.responses import Response
        return Response(
            content=thumbnail_data,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={translation_id}.jpg"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving thumbnail: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve thumbnail: {str(e)}"
        )


@app.get(
    "/user/{user_id}/t2v/recent-translations",
    summary="Get recent translations",
    description="Retrieves the 3 most recent translations for a user"
)
async def get_recent_t2v_translations(user_id: int):
    try:
        # Get user directory
        user_dir, _, _, _, _ = ensure_user_directory(user_id)
        recent_dir = os.path.join(user_dir, "recent_translations")
        recent_file = os.path.join(recent_dir, "recent.json")

        if not os.path.exists(recent_file):
            return {"recent_translations": []}

        try:
            with open(recent_file, 'r') as f:
                recent_translations = json.load(f)

            return {
                "recent_translations": recent_translations,
                "count": len(recent_translations)
            }
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in recent translations file for user {user_id}")
            return {"recent_translations": []}

    except Exception as e:
        logger.error(f"Error retrieving recent translations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve recent translations: {str(e)}"
        )


@app.get("/user/{user_id}/recent-translations/v2t/thumbnail/{thumbnail_id}")
async def get_recent_v2t_thumbnail(user_id: int, thumbnail_id: str):
    """
    Get thumbnail for a recent video-to-text translation

    Args:
        user_id: ID of the user
        thumbnail_id: ID of the thumbnail

    Returns:
        Thumbnail image
    """
    thumbnail_path = STORAGE_ROOT / "users" / str(user_id) / "recent_translations" / "thumbnails" / f"{thumbnail_id}.jpg"

    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="V2T thumbnail not found")

    try:
        with open(thumbnail_path, "rb") as f:
            content = f.read()

        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={thumbnail_id}.jpg"}
        )
    except Exception as e:
        logger.error(f"Error retrieving v2t thumbnail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve thumbnail: {str(e)}")


@app.get("/user/{user_id}/recent-translations/t2v/thumbnail/{thumbnail_id}")
async def get_recent_t2v_thumbnail(user_id: int, thumbnail_id: str):
    """
    Get thumbnail for a recent text-to-video translation

    Args:
        user_id: ID of the user
        thumbnail_id: ID of the thumbnail

    Returns:
        Thumbnail image
    """
    # Get user directory and path to t2v thumbnails
    user_dir, _, _, _, _ = ensure_user_directory(user_id)
    thumbnail_path = pathlib.Path(os.path.join(user_dir, "recent_translations", "thumbnails", f"{thumbnail_id}.jpg"))

    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="T2V thumbnail not found")

    try:
        with open(thumbnail_path, "rb") as f:
            content = f.read()

        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={thumbnail_id}.jpg"}
        )
    except Exception as e:
        logger.error(f"Error retrieving t2v thumbnail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve thumbnail: {str(e)}")


@app.get(
    "/user/{user_id}/t2v/audio/{audio_id}",
    summary="Get audio for a text-to-sign translation",
    description="Retrieves the audio file for a specific text-to-sign translation"
)
async def get_t2v_audio(user_id: int, audio_id: str):
    try:
        # Get user directory structure
        user_dir, _, _, audio_dir, _ = ensure_user_directory(user_id)

        # Check if it's a temp audio or a saved audio
        audio_path = None
        if audio_id.isdigit():
            # It's a saved audio
            audio_path = os.path.join(audio_dir, f"{audio_id}.mp3")
        else:
            # It's a temp audio
            temp_files = [f for f in os.listdir(audio_dir) if f.startswith("temp_") and f.endswith(".mp3")]
            for temp_file in temp_files:
                if temp_file.startswith(f"temp_{audio_id}") or temp_file == f"temp_{audio_id}.mp3":
                    audio_path = os.path.join(audio_dir, temp_file)
                    break

        if not audio_path or not os.path.exists(audio_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audio not found"
            )

        # Read the audio file
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        # Return the audio with proper headers
        from fastapi.responses import Response
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"inline; filename={os.path.basename(audio_path)}"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audio: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve audio: {str(e)}"
        )

@app.get(
    "/user/{user_id}/t2v/recent-saves",
    summary="Get recent saved translations",
    description="Retrieves the 3 most recently saved translations for a user"
)
async def get_recent_t2v_saves(user_id: int, request: Request = None):
    try:
        # Get user directory structure
        user_dir, base64_dir, thumbnails_dir, audio_dir, metadata_path = ensure_user_directory(user_id)

        # Get metadata
        metadata = get_user_metadata(metadata_path)

        # Sort translations by saved_at timestamp (newest first)
        sorted_translations = sorted(
            metadata["translations"],
            key=lambda x: datetime.strptime(x.get("saved_at", "1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"),
            reverse=True
        )

        # Take only the 3 most recent
        recent_translations = sorted_translations[:3]

        # Load each translation file with details
        result = []
        for item in recent_translations:
            try:
                file_path = os.path.join(base64_dir, item["filename"])
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        translation_data = json.load(f)

                    # Get thumbnail URL if available
                    thumbnail_url = None
                    thumbnail_path = os.path.join(thumbnails_dir, f"{item['translation_id']}.jpg")
                    if os.path.exists(thumbnail_path) and request:
                        thumbnail_url = f"{request.base_url}user/{user_id}/t2v/thumbnail/{item['translation_id']}"

                    # Get audio URL if available
                    audio_url = None
                    audio_path = os.path.join(audio_dir, f"{item['translation_id']}.mp3")
                    if os.path.exists(audio_path) and request:
                        audio_url = f"{request.base_url}user/{user_id}/t2v/audio/{item['translation_id']}"

                    result.append({
                        "translation_id": item["translation_id"],
                        "original_text": item["original_text"],
                        "words": item["words"],
                        "saved_at": item["saved_at"],
                        "thumbnail_url": thumbnail_url,
                        "audio_url": audio_url,
                        "has_audio": item.get("has_audio", False),
                        "has_thumbnail": item.get("has_thumbnail", False),
                         "img": translation_data["img"]
                    })
                else:
                    logger.warning(f"Translation file missing: {file_path}")
            except Exception as e:
                logger.error(f"Error loading translation {item['translation_id']}: {e}")

        return {
            "recent_saves": result,
            "count": len(result)
        }

    except Exception as e:
        logger.error(f"Error retrieving recent saves: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve recent saves: {str(e)}"
        )

@app.get(
    "/user/{user_id}/search/t2v",
    summary="Search saved translations",
    description="Searches for saved translations containing the query text"
)
async def search_t2v_translations(
    user_id: int,
    query: str = Query(..., description="Text to search for"),
    request: Request = None
):
    try:
        if not query or query.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )

        # Get user directory structure
        user_dir, base64_dir, thumbnails_dir, audio_dir, metadata_path = ensure_user_directory(user_id)

        # Get metadata
        metadata = get_user_metadata(metadata_path)

        # Initialize results
        results = []

        # Case-insensitive search
        query = query.lower()

        # Search through metadata translations
        for item in metadata["translations"]:
            # Check if query is in the original text
            original_text = item["original_text"].lower()

            if query in original_text:
                # Found a match, load the translation data
                try:
                    file_path = os.path.join(base64_dir, item["filename"])
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            translation_data = json.load(f)

                        # Get thumbnail URL if available
                        thumbnail_url = None
                        thumbnail_path = os.path.join(thumbnails_dir, f"{item['translation_id']}.jpg")
                        if os.path.exists(thumbnail_path) and request:
                            thumbnail_url = f"{request.base_url}user/{user_id}/t2v/thumbnail/{item['translation_id']}"

                        # Get audio URL if available
                        audio_url = None
                        audio_path = os.path.join(audio_dir, f"{item['translation_id']}.mp3")
                        if os.path.exists(audio_path) and request:
                            audio_url = f"{request.base_url}user/{user_id}/t2v/audio/{item['translation_id']}"

                        # Create highlighted context
                        start_pos = original_text.find(query)
                        if start_pos >= 0:
                            # Get some context around the match
                            context_start = max(0, start_pos - 30)
                            context_end = min(len(original_text), start_pos + len(query) + 30)

                            # Extract the context
                            if context_start > 0:
                                prefix = "..." + item["original_text"][context_start:start_pos]
                            else:
                                prefix = item["original_text"][:start_pos]

                            if context_end < len(original_text):
                                suffix = item["original_text"][start_pos + len(query):context_end] + "..."
                            else:
                                suffix = item["original_text"][start_pos + len(query):]

                            # The matched text
                            matched = item["original_text"][start_pos:start_pos + len(query)]

                            # Create the highlighted context
                            match_context = f"{prefix}<highlight>{matched}</highlight>{suffix}"
                        else:
                            match_context = item["original_text"]

                        # Create result object
                        result = {
                            "translation_id": item["translation_id"],
                            "original_text": item["original_text"],
                            "match_context": match_context,
                            "words": item["words"],
                            "saved_at": item["saved_at"],
                            "thumbnail_url": thumbnail_url,
                            "audio_url": audio_url,
                            "has_audio": item.get("has_audio", False),
                            "has_thumbnail": item.get("has_thumbnail", False),
                            "img": translation_data["img"]
                        }

                        results.append(result)

                except Exception as e:
                    logger.error(f"Error loading translation {item['translation_id']} during search: {e}")

        # Sort results by saved_at timestamp (newest first)
        results.sort(key=lambda x: datetime.strptime(x.get("saved_at", "1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S"), reverse=True)

        return {
            "results": results,
            "total": len(results),
            "query": query
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching translations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search translations: {str(e)}"
        )




@app.get(
    "/user/{user_id}/search",
    summary="Search all translations",
    description="Searches for both text-to-sign and video-to-text translations containing the query text"
)
async def search_all_translations(
    user_id: int,
    query: str = Query(..., description="Text to search for"),
    request: Request = None
):
    try:
        if not query or query.strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )

        # Get results from both search functions
        t2v_results = []
        v2t_results = []

        # Get text-to-video results
        try:
            t2v_response = await search_t2v_translations(user_id, query, request)
            t2v_results = t2v_response.get("results", [])

            # Add source type to each result
            for result in t2v_results:
                result["source_type"] = "text-to-sign"

                # Normalize result keys for consistency with v2t results
                if "translation_id" in result and "id" not in result:
                    result["id"] = result["translation_id"]

        except Exception as e:
            logger.error(f"Error searching text-to-sign translations: {e}", exc_info=True)
            t2v_results = []

        # Get video-to-text results
        try:
            v2t_response = await search_v2t_translations(user_id, query, request)
            v2t_results = v2t_response.get("results", [])

            # Add source type to each result
            for result in v2t_results:
                result["source_type"] = "video-to-text"

                # Normalize result keys for consistency with t2v results
                if "id" in result and "translation_id" not in result:
                    result["translation_id"] = result["id"]

        except Exception as e:
            logger.error(f"Error searching video-to-text translations: {e}", exc_info=True)
            v2t_results = []

        # Combine results
        all_results = t2v_results + v2t_results

        # Function to get a datetime object from various timestamp formats
        def get_timestamp(item):
            timestamp = None

            # Try different timestamp fields in order of preference
            if "saved_at" in item:
                timestamp = item["saved_at"]
            elif "timestamp" in item:
                timestamp = item["timestamp"]
            else:
                return datetime(1970, 1, 1)  # Default to oldest date if no timestamp

            # Handle different timestamp formats
            try:
                if isinstance(timestamp, str):
                    if "T" in timestamp:  # ISO format
                        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    else:
                        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, (int, float)):
                    return datetime.fromtimestamp(timestamp)
                else:
                    return datetime(1970, 1, 1)
            except (ValueError, TypeError):
                return datetime(1970, 1, 1)

        # Sort combined results by timestamp (newest first)
        all_results.sort(key=get_timestamp, reverse=True)

        # Return combined results
        return {
            "results": all_results,
            "total": len(all_results),
            "query": query,
            "sources": {
                "text_to_sign": len(t2v_results),
                "video_to_text": len(v2t_results)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching all translations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search translations: {str(e)}"
        )



@app.get(
    "/user/{user_id}/recent-translations/all",
    summary="Get all recent translations",
    description="Retrieves recent translations from both video-to-text and text-to-sign systems"
)
async def get_all_recent_translations(user_id: int, request: Request = None):
    try:
        # Initialize results containers
        all_translations = []

        # Get video-to-text recent translations
        try:
            v2t_result = await get_recent_v2t_translations(user_id)
            v2t_translations = v2t_result.get("translations", [])

            # Add type information to each translation
            for translation in v2t_translations:
                translation["source_type"] = "video-to-text"

                # Add video URL if available
                if "id" in translation:
                    translation_id = translation["id"]
                    video_path = STORAGE_ROOT / "users" / str(user_id) / "recent_translations" / "videos" / f"{translation_id}.mp4"

                    if video_path.exists() and request:
                        translation["video_url"] = f"{request.base_url}user/{user_id}/recent-translations/video/{translation_id}"

                # Add thumbnail URL if available
                if "id" in translation and request:
                    translation_id = translation["id"]
                    thumbnail_path = STORAGE_ROOT / "users" / str(user_id) / "recent_translations" / "thumbnails" / f"{translation_id}.jpg"

                    if thumbnail_path.exists():
                        translation["thumbnail_url"] = f"{request.base_url}user/{user_id}/recent-translations/v2t/thumbnail/{translation_id}"

            all_translations.extend(v2t_translations)

        except Exception as e:
            print(f"Error getting v2t recent translations: {str(e)}")

        # Get text-to-sign recent translations
        try:
            t2v_result = await get_recent_t2v_translations(user_id)
            t2v_translations = t2v_result.get("recent_translations", [])

            # Add type information to each translation
            for translation in t2v_translations:
                translation["source_type"] = "text-to-sign"

                # Normalize field names if needed
                if "id" in translation and "translation_id" not in translation:
                    translation["translation_id"] = translation["id"]

                # Make sure we have timestamp in a consistent format
                if "timestamp" in translation and "datetime" not in translation.get("timestamp", ""):
                    # Convert ISO format to datetime format if needed
                    try:
                        dt = datetime.fromisoformat(translation["timestamp"].replace("Z", "+00:00"))
                        translation["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass

            all_translations.extend(t2v_translations)

        except Exception as e:
            print(f"Error getting t2v recent translations: {str(e)}")

        # Sort all translations by timestamp (newest first)
        def get_timestamp(item):
            # Try different timestamp fields
            timestamp = item.get("timestamp", "")

            # Try to parse the timestamp to enable sorting
            try:
                if "T" in timestamp:  # ISO format
                    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except:
                # Return oldest possible date if parsing fails
                return datetime(1970, 1, 1)

        all_translations.sort(key=get_timestamp, reverse=True)

        # Return combined results
        return {
            "translations": all_translations,
            "count": len(all_translations),
            "sources": {
                "video_to_text": sum(1 for t in all_translations if t.get("source_type") == "video-to-text"),
                "text_to_sign": sum(1 for t in all_translations if t.get("source_type") == "text-to-sign")
            }
        }

    except Exception as e:
        error_msg = f"Error retrieving all recent translations: {str(e)}"
        print(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )



@app.get(
    "/user/{user_id}/recent-saves/all",
    summary="Get all recent saved translations",
    description="Retrieves recent saved translations from both video-to-text and text-to-sign systems"
)
async def get_all_recent_saves(user_id: int, request: Request):
    try:
        # Initialize results containers
        all_saves = []

        # Get video-to-text recent saves
        try:
            v2t_result = await get_recent_v2t_saves(user_id, request)
            v2t_saves = v2t_result.get("saves", [])

            # Add type information to each save
            for save in v2t_saves:
                save["source_type"] = "video-to-text"

                # Normalize field names
                if "id" in save and "translation_id" not in save:
                    save["translation_id"] = save["id"]

                # Ensure timestamp is in a consistent format
                if "timestamp" in save:
                    # Convert ISO format to datetime format if needed
                    try:
                        if isinstance(save["timestamp"], str) and "T" in save["timestamp"]:
                            dt = datetime.fromisoformat(save["timestamp"].replace("Z", "+00:00"))
                            save["timestamp_formatted"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass

            all_saves.extend(v2t_saves)

        except Exception as e:
            logger.error(f"Error getting v2t recent saves: {e}", exc_info=True)

        # Get text-to-sign recent saves
        try:
            t2v_result = await get_recent_t2v_saves(user_id, request)
            t2v_saves = t2v_result.get("recent_saves", [])

            # Add type information to each save
            for save in t2v_saves:
                save["source_type"] = "text-to-sign"

                # Normalize field names
                if "translation_id" in save and "id" not in save:
                    save["id"] = save["translation_id"]

                # Rename saved_at to timestamp for consistency
                if "saved_at" in save and "timestamp" not in save:
                    save["timestamp"] = save["saved_at"]
                    # Also store formatted version for sorting
                    try:
                        dt = datetime.strptime(save["saved_at"], "%Y-%m-%d %H:%M:%S")
                        save["timestamp_formatted"] = save["saved_at"]
                    except:
                        pass

            all_saves.extend(t2v_saves)

        except Exception as e:
            logger.error(f"Error getting t2v recent saves: {e}", exc_info=True)

        # Sort all saves by timestamp (newest first)
        def get_timestamp(item):
            # Try different timestamp fields in order of preference
            if "timestamp_formatted" in item:
                try:
                    return datetime.strptime(item["timestamp_formatted"], "%Y-%m-%d %H:%M:%S")
                except:
                    pass

            if "timestamp" in item:
                timestamp = item["timestamp"]
                try:
                    if isinstance(timestamp, str):
                        if "T" in timestamp:  # ISO format
                            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        else:
                            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    elif isinstance(timestamp, (int, float)):
                        return datetime.fromtimestamp(timestamp)
                except:
                    pass

            if "saved_at" in item:
                try:
                    return datetime.strptime(item["saved_at"], "%Y-%m-%d %H:%M:%S")
                except:
                    pass

            # Return oldest possible date if all parsing fails
            return datetime(1970, 1, 1)

        all_saves.sort(key=get_timestamp, reverse=True)

        # Return combined results
        return {
            "saves": all_saves,
            "count": len(all_saves),
            "sources": {
                "video_to_text": sum(1 for s in all_saves if s.get("source_type") == "video-to-text"),
                "text_to_sign": sum(1 for s in all_saves if s.get("source_type") == "text-to-sign")
            }
        }

    except Exception as e:
        error_msg = f"Error retrieving all recent saves: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )




@app.get("/user/{user_id}/saved/all")
async def get_all_saved_translations(user_id: int, request: Request):
    """
    Get all saved translations from both text-to-video and video-to-text systems

    Args:
        user_id: ID of the user
        request: Request object to get base URL

    Returns:
        Combined list of all saved translations sorted by timestamp
    """
    try:
        # Initialize combined results list
        all_translations = []
        
        # 1. Get V2T translations with text
        try:
            v2t_text_result = await get_translations_with_text(user_id, request)
            v2t_text_translations = v2t_text_result.get("translations", [])
            
            # Add source type and ensure consistent field names
            for translation in v2t_text_translations:
                translation["source_type"] = "video-to-text"
                translation["translation_type"] = "text"
                
                # Ensure consistent field names
                if "id" in translation and "translation_id" not in translation:
                    translation["translation_id"] = translation["id"]
                
            all_translations.extend(v2t_text_translations)
        except Exception as e:
            print(f"Error fetching V2T text translations: {e}")
        
        # 2. Get V2T translations with audio
        try:
            v2t_audio_result = await get_translations_with_audio(user_id, request)
            v2t_audio_translations = v2t_audio_result.get("translations", [])
            
            # Filter out any translations already added from text endpoint
            existing_ids = {t.get("translation_id", t.get("id")) for t in all_translations}
            v2t_audio_translations = [t for t in v2t_audio_translations 
                                     if t.get("id") not in existing_ids]
            
            # Add source type and ensure consistent field names
            for translation in v2t_audio_translations:
                translation["source_type"] = "video-to-text"
                translation["translation_type"] = "audio"
                
                # Ensure consistent field names
                if "id" in translation and "translation_id" not in translation:
                    translation["translation_id"] = translation["id"]
                
            all_translations.extend(v2t_audio_translations)
        except Exception as e:
            print(f"Error fetching V2T audio translations: {e}")
        
        # 3. Get T2V translations
        try:
            t2v_result = await get_all_translations(user_id, request)
            t2v_translations = t2v_result.get("translations", [])
            
            # Add source type and ensure consistent field names
            for translation in t2v_translations:
                translation["source_type"] = "text-to-video"
                translation["translation_type"] = "animation"
                
                # Rename fields for consistency
                if "saved_at" in translation and "timestamp" not in translation:
                    translation["timestamp"] = translation["saved_at"]
                
                if "original_text" in translation and "text" not in translation:
                    translation["text"] = translation["original_text"]
                
                # Ensure ID fields exist in both formats
                if "translation_id" in translation and "id" not in translation:
                    translation["id"] = translation["translation_id"]
                
            all_translations.extend(t2v_translations)
        except Exception as e:
            print(f"Error fetching T2V translations: {e}")
        
        # Sort all translations by timestamp (newest first) - using the same logic from recent-saves/all
        def get_timestamp(item):
            # Try different timestamp fields in order of preference
            if "timestamp_formatted" in item:
                try:
                    return datetime.strptime(item["timestamp_formatted"], "%Y-%m-%d %H:%M:%S")
                except:
                    pass

            if "timestamp" in item:
                timestamp = item["timestamp"]
                try:
                    if isinstance(timestamp, str):
                        if "T" in timestamp:  # ISO format
                            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        else:
                            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    elif isinstance(timestamp, (int, float)):
                        return datetime.fromtimestamp(timestamp)
                except:
                    pass

            if "saved_at" in item:
                try:
                    return datetime.strptime(item["saved_at"], "%Y-%m-%d %H:%M:%S")
                except:
                    pass

            # Return oldest possible date if all parsing fails
            return datetime(1970, 1, 1)
            
        all_translations.sort(key=get_timestamp, reverse=True)
        
        # Return the combined results
        return {
            "translations": all_translations,
            "count": len(all_translations),
            "sources": {
                "video_to_text": sum(1 for t in all_translations if t.get("source_type") == "video-to-text"),
                "text_to_video": sum(1 for t in all_translations if t.get("source_type") == "text-to-video")
            }
        }
        
    except Exception as e:
        print(f"Error retrieving all saved translations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve all saved translations: {str(e)}"
        )
    


@app.websocket("/ws/{device_type}/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_type: str, device_id: str):
    await manager.connect(websocket, device_id, device_type)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "broadcast":
                # If message is from mobile, convert to sign language instead of broadcasting normally
                if device_type == "mobile":
                    text_message = message_data.get("message", "").strip()
                    if text_message:
                        # Start text-to-sign conversion in background
                        asyncio.create_task(
                            process_text_to_sign(
                                text=text_message,
                                client_id=device_id,
                                device_type=device_type
                            )
                        )
                else:
                    # For non-mobile devices, broadcast normally
                    await manager.broadcast_message(
                        json.dumps({
                            "type": "message",
                            "from": device_id,
                            "device_type": device_type,
                            "message": message_data.get("message"),
                            "timestamp": message_data.get("timestamp")
                        }),
                        sender_id=device_id
                    )
            elif message_data.get("type") == "direct":
                # If message is from mobile, convert to sign language instead of direct messaging
                if device_type == "mobile":
                    text_message = message_data.get("message", "").strip()
                    if text_message:
                        # Start text-to-sign conversion in background
                        asyncio.create_task(
                            process_text_to_sign(
                                text=text_message,
                                client_id=device_id,
                                device_type=device_type
                            )
                        )
                else:
                    # For non-mobile devices, send direct message normally
                    target_device = message_data.get("target_device")
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "direct_message",
                            "from": device_id,
                            "device_type": device_type,
                            "message": message_data.get("message"),
                            "timestamp": message_data.get("timestamp")
                        }),
                        target_device
                    )
            elif message_data.get("type") == "get_devices":
                device_list = [
                    {"device_id": did, "type": info["type"]} 
                    for did, info in manager.device_info.items()
                    if did != device_id
                ]
                await manager.send_personal_message(
                    json.dumps({
                        "type": "device_list",
                        "devices": device_list
                    }),
                    device_id
                )
            
    except WebSocketDisconnect:
        # Clean disconnect - just remove from manager
        manager.disconnect(device_id)
        # Notify others about disconnect (safely)
        try:
            await manager.broadcast_system_message(f"{device_type} device {device_id} disconnected")
        except Exception as e:
            print(f"Error broadcasting disconnect message: {e}")
    except Exception as e:
        print(f"Unexpected error in websocket for {device_id}: {e}")
        manager.disconnect(device_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "connected_devices": len(manager.active_connections)}

@app.get("/devices")
async def get_devices():
    return {
        "devices": [
            {"device_id": device_id, "type": info["type"]} 
            for device_id, info in manager.device_info.items()
        ]
    }


@app.post("/upload-device/")
async def upload_video_device(
    file: UploadFile = File(...),
    client_id: str = Form(...)
):
    """
    Upload a video file for translation from device communication system
    """
    # Check if client is connected via WebSocket
    if client_id not in manager.active_connections:
        raise HTTPException(
            status_code=400,
            detail="Client not connected via WebSocket. Connect first"
        )
    
    # Check if it's a video file
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail="Please upload a video file"
        )
    
    # Create a temporary file for the video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        # Write the uploaded file to the temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        
        original_filename = file.filename
        
        # Start processing in a background task
        asyncio.create_task(
            process_video_device(
                video_path=temp_file.name,
                client_id=client_id,
                original_filename=original_filename
            )
        )
        
        # Return success response
        return {"message": "Video upload successful, processing started"}
        
    except Exception as e:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        # Send error notification via WebSocket
        try:
            if client_id in manager.active_connections:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "video_processing",
                        "status": "error",
                        "message": f"Error uploading video: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }),
                    client_id
                )
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing upload: {str(e)}"
        )
    


@app.get("/bridge-conversation")
async def get_bridge_conversation():
    """Get the bridge conversation for mobile app"""
    bridge_file = "bridge_conversations/conversation.json"
    
    if not os.path.exists(bridge_file):
        return {
            "title": "Sign Language Bridge Conversation",
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    with open(bridge_file, 'r') as f:
        conversation_data = json.load(f)
    
    return conversation_data


@app.get("/bridge-conversation/recent")
async def get_recent_bridge_messages():
    """Get the last 2 messages from the bridge conversation"""
    bridge_file = "bridge_conversations/conversation.json"
    
    if not os.path.exists(bridge_file):
        return {
            "title": "Sign Language Bridge Conversation",
            "messages": []
        }
    
    try:
        with open(bridge_file, 'r') as f:
            conversation_data = json.load(f)
        
        # Get the last 2 messages
        all_messages = conversation_data.get("messages", [])
        recent_messages = all_messages[-2:] if len(all_messages) >= 2 else all_messages
        
        # Extract only the required fields
        formatted_messages = []
        for msg in recent_messages:
            formatted_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("timestamp")
            })
        
        return {
            "title": conversation_data.get("title", "Sign Language Bridge Conversation"),
            "messages": formatted_messages
        }
        
    except Exception as e:
        logger.error(f"Error reading bridge conversation: {e}")
        return {
            "title": "Sign Language Bridge Conversation",
            "messages": []
        }