"""
Author: Jacob Pitsenberger
Program: main.py
Version: 1.0
Project: Detecting Filtered Classes with YOLOv8 Pretrained Model
Date: 12/5/2023
Purpose: This program contains the main method for initializing the filtered detector and using it to detect
         filtered objects over real-time video feeds, image files, or video files.
Uses: filtered_detector.py
"""

from filtered_detector import FilteredDetector

def main():
    """
    Main method to run the Detecting Filtered Classes with YOLOv8 Pretrained Model System.

    Returns:
        None
    """
    # Filter list for test image file.
    image_test_filters = ['car', 'truck']

    # Filter list for test video file.
    video_test_filters = ['chair', 'couch', 'potted plant', 'dining table', 'tv']

    # Filter list for testing over internal webcam realtime feed.
    realtime_test_filters = ['person', 'bottle', 'backpack', 'spoon', 'knife']

    # Initialize the detector with the desired list of object names to filter detections for.
    detector = FilteredDetector(image_test_filters)

    # Call one of the detectors methods for processing image files, video files, or realtime streams for detections.
    detector.detect_over_image('test_files/test_image.jpg')
    # detector.detect_over_video_file('test_files/test_video.mp4')
    # detector.detect_over_realtime_feed()


if __name__ == "__main__":
    main()
