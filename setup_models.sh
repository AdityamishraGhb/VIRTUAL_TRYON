#!/bin/bash

# Virtual Saree Try-On - Model Setup Script
# This script downloads all required AI models

set -e

echo "🚀 Setting up Virtual Saree Try-On models..."

# Create models directory
mkdir -p models
cd models

echo "📦 Downloading MODNet model..."
if [ ! -f "modnet_photographic_portrait_matting.onnx" ]; then
    wget -O modnet_photographic_portrait_matting.onnx \
        "https://github.com/ZHKKKe/MODNet/releases/download/pretrained_ckpt/modnet_photographic_portrait_matting.onnx"
    echo "✅ MODNet model downloaded"
else
    echo "✅ MODNet model already exists"
fi

echo "📦 Downloading Real-ESRGAN model..."
if [ ! -f "RealESRGAN_x4plus.pth" ]; then
    wget -O RealESRGAN_x4plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    echo "✅ Real-ESRGAN model downloaded"
else
    echo "✅ Real-ESRGAN model already exists"
fi

echo "📦 Downloading MediaPipe Pose model..."
if [ ! -f "pose_landmarker.task" ]; then
    wget -O pose_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    echo "✅ MediaPipe Pose model downloaded"
else
    echo "✅ MediaPipe Pose model already exists"
fi

# Note: VITON-HD model needs to be obtained separately
echo "⚠️  VITON-HD model needs to be downloaded separately"
echo "   Please visit: https://github.com/shadow2496/VITON-HD"
echo "   And place the model file as: viton_hd.pth"

cd ..

echo "🎉 Model setup completed!"
echo ""
echo "📁 Models directory structure:"
ls -la models/
echo ""
echo "🚀 You can now run the application with: python main.py"