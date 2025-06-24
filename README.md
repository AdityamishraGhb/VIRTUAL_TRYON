# Virtual Saree Try-On Backend

A complete FastAPI backend system for AI-powered virtual saree fitting using MODNet, MediaPipe, VITON-HD, and Real-ESRGAN.

## 🚀 Features

- **Background Removal**: MODNet-based background removal for clean garment isolation
- **Pose Detection**: MediaPipe pose estimation for accurate body keypoint detection
- **Garment Warping**: VITON-HD approach for realistic saree draping simulation
- **Image Enhancement**: Real-ESRGAN super-resolution for high-quality output
- **RESTful API**: Complete FastAPI backend with async processing
- **File Management**: Organized storage with unique job IDs
- **Status Tracking**: Real-time processing status monitoring

## 📁 Project Structure

```
virtual_tryon/
├── main.py                 # FastAPI application entry point
├── upload.py              # Image upload and processing pipeline
├── remove_bg.py           # MODNet background removal
├── pose_detect.py         # MediaPipe pose detection
├── warp.py                # VITON-HD saree warping
├── enhance.py             # Real-ESRGAN image enhancement
├── preview.py             # Results preview and download
├── utils/
│   ├── modnet_utils.py    # MODNet utility functions
│   ├── mediapipe_utils.py # MediaPipe utility functions
│   └── esrgan_utils.py    # Real-ESRGAN utility functions
├── data/                  # Job storage directory
│   └── {job_id}/         # Individual job folders
├── models/               # AI model storage
└── requirements.txt      # Python dependencies
```

## 🛠️ Installation

1. **Clone and Setup**:
```bash
cd virtual_tryon
pip install -r requirements.txt
```

2. **Download AI Models**:
```bash
# Create models directory
mkdir -p models

# Download MODNet model
wget -O models/modnet_photographic_portrait_matting.onnx \
  "https://github.com/ZHKKKe/MODNet/releases/download/pretrained_ckpt/modnet_photographic_portrait_matting.onnx"

# Download Real-ESRGAN model
wget -O models/RealESRGAN_x4plus.pth \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Download VITON-HD model (if available)
# wget -O models/viton_hd.pth "VITON_HD_MODEL_URL"
```

3. **Run the Server**:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## 📡 API Endpoints

### Upload Images
```http
POST /api/v1/upload
Content-Type: multipart/form-data

body: file (image)
pallu: file (image)
blouse: file (image)
```

**Response**:
```json
{
  "message": "Images uploaded successfully. Processing started.",
  "job_id": "uuid-string",
  "status": "processing",
  "estimated_time": "2-3 minutes"
}
```

### Check Status
```http
GET /api/v1/status/{job_id}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "output_url": "/static/uuid-string/final_output.jpg",
  "preview_url": "/api/v1/preview/uuid-string"
}
```

### Get Preview
```http
GET /api/v1/preview/{job_id}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "files": {
    "body_nobg.jpg": "/static/uuid-string/body_nobg.jpg",
    "final_output.jpg": "/static/uuid-string/final_output.jpg"
  },
  "pose_data": { ... },
  "final_output": "/static/uuid-string/final_output.jpg",
  "thumbnail": "/static/uuid-string/thumbnail.jpg"
}
```

### Download Result
```http
GET /api/v1/download/{job_id}
```

Returns the final processed image file.

### List Jobs
```http
GET /api/v1/jobs?limit=10&offset=0
```

### Get Statistics
```http
GET /api/v1/stats
```

## 🔄 Processing Pipeline

1. **Upload**: User uploads 3 images (body, pallu, blouse)
2. **Background Removal**: MODNet removes backgrounds from all images
3. **Pose Detection**: MediaPipe extracts body keypoints from body image
4. **Garment Warping**: VITON-HD warps saree components onto the body
5. **Enhancement**: Real-ESRGAN upscales and enhances the final image
6. **Storage**: Results saved with unique job ID for retrieval

## 🎯 Model Integration

### MODNet (Background Removal)
- **Model**: `modnet_photographic_portrait_matting.onnx`
- **Input**: RGB image (512x512)
- **Output**: Alpha matte for background removal

### MediaPipe (Pose Detection)
- **Model**: Built-in MediaPipe Pose
- **Input**: RGB image
- **Output**: 33 body keypoints with confidence scores

### VITON-HD (Garment Warping)
- **Model**: `viton_hd.pth` (PyTorch)
- **Input**: Body image, garment images, pose keypoints
- **Output**: Warped garment on body

### Real-ESRGAN (Super Resolution)
- **Model**: `RealESRGAN_x4plus.pth`
- **Input**: Low-resolution image
- **Output**: 4x upscaled high-quality image

## 🔧 Configuration

### Environment Variables
```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Model Paths
MODNET_MODEL_PATH=models/modnet_photographic_portrait_matting.onnx
ESRGAN_MODEL_PATH=models/RealESRGAN_x4plus.pth
VITON_MODEL_PATH=models/viton_hd.pth

# Storage
DATA_DIR=data
MODELS_DIR=models
```

### Model Fallbacks
The system includes fallback methods when AI models are not available:
- **MODNet**: Color-based background removal
- **MediaPipe**: Geometric pose estimation
- **VITON-HD**: Perspective transformation warping
- **Real-ESRGAN**: Bicubic upscaling with sharpening

## 🧪 Testing

```bash
# Run tests
pytest

# Test with sample images
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "body=@sample_body.jpg" \
  -F "pallu=@sample_pallu.jpg" \
  -F "blouse=@sample_blouse.jpg"
```

## 📊 Performance

- **Processing Time**: 2-3 minutes per job (with GPU acceleration)
- **Memory Usage**: ~4GB RAM for full pipeline
- **Storage**: ~50MB per processed job
- **Concurrent Jobs**: Supports multiple simultaneous processing

## 🔒 Security

- File type validation (JPEG/PNG only)
- File size limits
- Unique job IDs prevent conflicts
- Automatic cleanup of old jobs
- CORS protection

## 🚀 Production Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Scaling
- Use Redis for job queue management
- Deploy with Kubernetes for auto-scaling
- Add load balancer for multiple instances
- Use cloud storage for model and data files

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📞 Support

For issues and questions:
- Create GitHub issue
- Check documentation
- Review API examples
# VIRTUAL_TRYON

