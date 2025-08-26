import torch
from matplotlib import pyplot as plt
from PIL import Image

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 's' is small/fast version

# List of image paths (you can change or expand this list)
image_paths = [
    'Image Detecting/images/image1.jpg',
    'Image Detecting/images/image2.jpg',
    'Image Detecting/images/image3.jpg'
]

# Loop through images
for img_path in image_paths:
    print(f"\n Detecting in: {img_path}")
    results = model(img_path)

    # Print detected classes
    results.print()

    # Show image with bounding boxes
    results.show()