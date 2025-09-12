# HandTalk 🤟
*Sign Language Translation System - Graduation Project*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🎓 Project Overview

HandTalk is a comprehensive sign language translation system that bridges the communication gap between deaf/hard-of-hearing individuals and hearing individuals. This graduation project implements a real-time, bidirectional translation system using state-of-the-art deep learning techniques.

### What HandTalk Does:
- **Video-to-Text**: Converts American Sign Language (ASL) videos into English text
- **Text-to-Sign**: Generates sign language animations from English text input
- **Real-time Processing**: Provides instant translations through a web/mobile interface
- **Multi-platform**: Works on web browsers, mobile devices, and desktop applications
- **Cloud-ready**: Runs entirely in Google Colab with free GPU access

### Technical Innovation:
- **Pose-only Recognition**: Uses efficient pose estimation instead of raw video processing
- **Uni-Sign Architecture**: Implements the latest sign language understanding model
- **WebSocket Communication**: Real-time bidirectional communication between users
- **RESTful API**: Complete backend system with multiple endpoints for different use cases

### Project Impact:
This system addresses the critical need for accessible communication tools in the deaf, hard-of-hearing community, and who rely on sign language as their primary means of communication.


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
