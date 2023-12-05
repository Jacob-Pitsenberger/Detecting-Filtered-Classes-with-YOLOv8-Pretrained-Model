# YOLOv8 Filtered Object Detection

## Overview

This project utilizes a YOLOv8 pretrained model from Ultralytics to perform filtered object detection on images, videos, or real-time webcam feeds. The filtered detector focuses on specific classes of objects from the COCO dataset. The included classes can be easily customized to suit your application.

## Prerequisites

- Python 3.x
- OpenCV
- Numpy
- Ultralytics YOLO

Install dependencies using:

```bash
pip install opencv-python numpy
pip install 'git+https://github.com/ultralytics/yolov5.git'
```
## Usage
Create a custom filter_classes list in the main.py file to specify the classes you want to detect. You can refer to the COCO dataset for a complete list of classes.

Example:
```bash
# Create a custom filter_classes list to include the classes you want to detect.
# You can refer to the COCO dataset for a complete list of classes: https://cocodataset.org/#explore
# Example classes: 'person', 'car'
filter_classes = ['person', 'car']
# More examples can be added: 'bird', 'dog', 'cat', 'bicycle', ...
```
Or utilize the defined lists used with the test files in this repository

Example:
```bash
image_test_filters = ['car', 'truck']
video_test_filters = ['chair', 'couch', 'potted plant', 'dining table', 'tv']
realtime_test_filters = ['cow', 'person', 'bottle', 'backpack', 'spoon', 'knife']
```

Initialize the FilteredDetector with the specified filter classes in the main.py file.

Example:
```bash
# Initialize the FilteredDetector with the specified filter classes
detector = FilteredDetector(filter_classes)
```

Uncomment the desired method in the main function to detect objects over an image file, video file, or real-time webcam feed.

Example:
```bash
# Uncomment one of the following lines to choose the detection method
# detector.detect_over_image('test_files/img.png')
# detector.detect_over_video_file('test_files/cows.mp4')
# detector.detect_over_realtime_feed()
```

Run the main.py file to see the filtered object detection in action.

## Notes
- This project uses a pretrained YOLOv8 model from Ultralytics, trained on the COCO dataset.

- Customize the filter_classes list to include the specific classes you want to detect.

- Feel free to explore and expand the functionality based on your project requirements.

## Author
Jacob Pitsenberger
December 5, 2023

## License

This software is licensed under the MIT License. By using this software, you agree to comply with the terms outlined in the license.