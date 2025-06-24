"""
Test script for Virtual Saree Try-On API
"""
import requests
import time
import json
from pathlib import Path


def test_api():
    """Test the complete API workflow"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Virtual Saree Try-On API...")
    
    # Test health check
    print("\n1. Testing health check...")
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.status_code} - {response.json()}")
    
    # Test upload (you'll need sample images)
    print("\n2. Testing image upload...")
    
    # Create dummy image files for testing
    sample_images = create_sample_images()
    
    files = {
        'body': ('body.jpg', sample_images['body'], 'image/jpeg'),
        'pallu': ('pallu.jpg', sample_images['pallu'], 'image/jpeg'),
        'blouse': ('blouse.jpg', sample_images['blouse'], 'image/jpeg')
    }
    
    response = requests.post(f"{base_url}/api/v1/upload", files=files)
    print(f"Upload response: {response.status_code}")
    
    if response.status_code == 202:
        upload_data = response.json()
        job_id = upload_data['job_id']
        print(f"Job ID: {job_id}")
        
        # Test status checking
        print("\n3. Testing status check...")
        for i in range(10):  # Check status for up to 10 times
            response = requests.get(f"{base_url}/api/v1/status/{job_id}")
            status_data = response.json()
            print(f"Status check {i+1}: {status_data['status']}")
            
            if status_data['status'] == 'completed':
                print(f"✅ Processing completed!")
                print(f"Output URL: {status_data.get('output_url')}")
                break
            elif status_data['status'].startswith('failed'):
                print(f"❌ Processing failed: {status_data['status']}")
                break
            
            time.sleep(5)  # Wait 5 seconds between checks
        
        # Test preview
        print("\n4. Testing preview...")
        response = requests.get(f"{base_url}/api/v1/preview/{job_id}")
        if response.status_code == 200:
            preview_data = response.json()
            print(f"Preview data: {json.dumps(preview_data, indent=2)}")
        
        # Test download
        print("\n5. Testing download...")
        response = requests.get(f"{base_url}/api/v1/download/{job_id}")
        if response.status_code == 200:
            with open(f"test_output_{job_id}.jpg", "wb") as f:
                f.write(response.content)
            print(f"✅ Downloaded result image: test_output_{job_id}.jpg")
    
    # Test job listing
    print("\n6. Testing job listing...")
    response = requests.get(f"{base_url}/api/v1/jobs")
    if response.status_code == 200:
        jobs_data = response.json()
        print(f"Total jobs: {jobs_data['total']}")
        print(f"Listed jobs: {len(jobs_data['jobs'])}")
    
    # Test statistics
    print("\n7. Testing statistics...")
    response = requests.get(f"{base_url}/api/v1/stats")
    if response.status_code == 200:
        stats_data = response.json()
        print(f"Statistics: {json.dumps(stats_data, indent=2)}")
    
    print("\n🎉 API testing completed!")


def create_sample_images():
    """Create sample images for testing"""
    import io
    from PIL import Image
    import numpy as np
    
    # Create simple colored rectangles as sample images
    def create_colored_image(color, size=(512, 512)):
        img_array = np.full((size[1], size[0], 3), color, dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    return {
        'body': create_colored_image([200, 180, 160]),    # Skin tone
        'pallu': create_colored_image([255, 100, 100]),   # Red saree
        'blouse': create_colored_image([100, 100, 255])   # Blue blouse
    }


def test_individual_endpoints():
    """Test individual endpoints"""
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/",
        "/health",
        "/api/v1/jobs",
        "/api/v1/stats"
    ]
    
    print("🔍 Testing individual endpoints...")
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            print(f"{endpoint}: {response.status_code} - {response.reason}")
        except Exception as e:
            print(f"{endpoint}: ERROR - {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting API tests...")
    print("Make sure the server is running on http://localhost:8000")
    print()
    
    try:
        test_individual_endpoints()
        print()
        test_api()
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")