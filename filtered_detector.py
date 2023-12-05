"""
Author: Jacob Pitsenberger
Program: filtered_detector.py
Version: 1.0
Project: Detecting Filtered Classes with YOLOv8 Pretrained Model
Date: 12/5/2023
Purpose: This module contains the FilteredDetector class, which is responsible for detecting and counting
         classes filtered from the coco dataset in a live video feed using the YOLO (You Only Look Once)
         v8 model. It implements the necessary functions to process realtime video frames or post process
         image or video files, apply Non-Maximum Suppression (NMS) to filter out duplicate detections,
         and draw bounding boxes around detected objects with confidence scores and class names on the image
         or video frames..
Uses: utils.py
"""
from utils import non_max_suppression, create_coco_classes_dict
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


class FilteredDetector:
    """
    A class for detecting and counting objects in a live video feed using YOLO V8 model.

    Attributes:
        FONT_SIZE (float): Font size for text display on the output window.
        DRAWING_COLOR (tuple): RGB color tuple for text and detection boxes drawn on the output window.
        DRAWING_THICKNESS (int): Thickness of text and detection boxes drawn on the output window.
    """

    FONT_SIZE = 0.7
    DRAWING_COLOR = (0, 255, 0)
    DRAWING_THICKNESS = 2

    def __init__(self, filter_classes: list) -> None:
        """
        Initialize the FilteredDetector object.
        """
        try:
            # Get the class names of the detections to filter for.
            self.filter_classes_names = filter_classes

            # Create a dictionary to use for class name to class id mapping.
            self.coco_classes_dict = create_coco_classes_dict()

            # Get the id values for the class names to filter detections for.
            self.filter_classes_ids = self.get_keys_for_values()

            # Set the confidence threshold (adjust as needed)
            self.confidence_threshold = 0.5

            # load a pretrained model (recommended for training)
            self.model = YOLO("yolov8n.pt")

            # Video capture will be initialized in methods
            self.cap = None

        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def get_keys_for_values(self) -> list:
        """
        Get the keys corresponding to the filter classes from the COCO classes dictionary.

        ----- Get the key (class ID) associated with a class name by iterating over the items in
              the dictionary, unpacking each key-value pair into the variables key and value
              and checking if the current value (class name) is present in the filtered class names list to
              return a list of the key (class IDs) associated with the class names to filter detections for.

        Returns:
            list: List of keys corresponding to the filter classes.
        """
        keys_for_values = [key for key, value in self.coco_classes_dict.items() if value in self.filter_classes_names]
        return keys_for_values

    def draw_detections(self, img, bbox, class_id, conf) -> np.ndarray:
        """
        Draw bounding boxes and class names on the image.

        Args:
            img (numpy.ndarray): Input image.
            bbox (list): List of bounding box coordinates.
            class_id (int): Class ID.
            conf (float): Confidence score.

        Returns:
            numpy.ndarray: Image with bounding boxes and class names drawn.
        """
        # Get the class name associated with the class id detected
        class_name = self.coco_classes_dict.get(class_id, None)

        # Draw the detection bounding box on the image
        img = cv2.rectangle(img, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), self.DRAWING_COLOR,
                            self.DRAWING_THICKNESS)

        # Draw the class name and confidence score on the image over the detection bounding box.
        img = cv2.putText(img, f'{class_name} - {conf:.2f}', (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_COMPLEX,
                          self.FONT_SIZE, self.DRAWING_COLOR, self.DRAWING_THICKNESS)

        # Return the image with drawn detection info on it.
        return img

    def detect_over_image(self, img_path: str) -> None:
        """
        Detect and draw bounding boxes on an image.

        Args:
            img_path (str): Path to the input image.

        Returns:
            None
        """
        try:
            # Specify the path to save the image with found detections.
            base_filename = Path(img_path).stem
            img_path_out = os.path.join('test_files', base_filename + '_predictions.png')

            # Load the model
            model = self.model

            # Read the image.
            img = cv2.imread(img_path)

            # Make detections on the image.
            results = model(img)[0]

            # Unwrap the detections so that we can get them in our desired format.
            for result in results:
                # Iterate through each result object treating it as a list.
                for r in result.boxes.data.tolist():
                    # Unwrap each class detected as its id, bounding box coordinates, and the confidence score for detecting.
                    x1, y1, x2, y2, score, class_id = r

                    # Filter Detections to only include those specified
                    if int(class_id) in self.filter_classes_ids:

                        # Get the bounding box coordinates of a detected object as a numpy array.
                        boxes = np.array([[int(x1), int(y1), int(x2), int(y2)]])

                        # Get the confidence score of the detected object as a numpy array wrapped in a list.
                        confidences = np.array([score])

                        # Apply Non-Maximum Suppression to filter out duplicate detections.
                        nms_indices = non_max_suppression(boxes, confidences)

                        # Iterate over the indices of the bounding boxes that have survived the Non-Maximum Suppression.
                        for idx in nms_indices:

                            # Get the confidence score associated with the current bounding box index.
                            confidence = confidences[idx]

                            # Check that the confidence score is greater than our specified threshold.
                            if confidence >= self.confidence_threshold:
                                # Get the bbox coordinates for the current index from the boxes array.
                                box = boxes[idx]

                                # Convert the coordinates of the bounding box to integer for drawing.
                                x1, y1, x2, y2 = box.astype(np.int_)

                                # Organize the coordinates into a list to pass to our drawing method.
                                bbox = [[x1, y1], [x2, y2]]

                                # Draw the bounding box and associated information on the image.
                                img = self.draw_detections(img, bbox, class_id, confidence)

            # Show the original image with all detections found.
            while True:
                cv2.imshow('image detection output', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Write this image with detections on it to our output path.
            cv2.imwrite(img_path_out, img)
            print(f"wrote img to the path: {img_path_out}")
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error detecting over image: {e}")
            raise

    def get_video_detections(self, results: list, frame: np.ndarray) -> np.ndarray:
        """
        Process video frames using YOLOv8 model.

        Args:
            results (list): Results from the YOLOv8 model.
            frame (numpy.ndarray): Current video frame in BGR format.

        Returns:
            numpy.ndarray: Processed video frame with bounding boxes drawn around objects.
        """
        try:
            # Unwrap the detections so that we can get them in our desired format.
            for result in results:

                # Iterate through each result object treating it as a list.
                for r in result.boxes.data.tolist():
                    # Unwrap each class detected as its id, bounding box coordinates, and the confidence score for detecting.
                    x1, y1, x2, y2, score, class_id = r

                    # Filter Detections to only include those specified
                    if int(class_id) in self.filter_classes_ids:

                        # Get the bounding box coordinates of a detected object as a numpy array.
                        boxes = np.array([[int(x1), int(y1), int(x2), int(y2)]])

                        # Get the confidence score of the detected object as a numpy array wrapped in a list.
                        confidences = np.array([score])

                        # Apply Non-Maximum Suppression to filter out duplicate detections.
                        nms_indices = non_max_suppression(boxes, confidences)

                        # Iterate over the indices of the bounding boxes that have survived the Non-Maximum Suppression.
                        for idx in nms_indices:

                            # Get the confidence score associated with the current bounding box index.
                            confidence = confidences[idx]

                            # Check that the confidence score is greater than our specified threshold.
                            if confidence >= self.confidence_threshold:
                                # Get the bbox coordinates for the current index from the boxes array.
                                box = boxes[idx]

                                # Convert the coordinates of the bounding box to integer for drawing.
                                x1, y1, x2, y2 = box.astype(np.int_)

                                # Organize the coordinates into a list to pass to our drawing method.
                                bbox = [[x1, y1], [x2, y2]]

                                # Draw the bounding box and associated information on the video frame.
                                frame = self.draw_detections(frame, bbox, class_id, confidence)

            # Return the frame with detections drawn on it.
            return frame
        except Exception as e:
            print(f"Error getting video file detection: {e}")
            raise

    def detect_over_video_file(self, video_path: str) -> None:
        """
        Process the live video feed and display the output.

        Args:
            video_path (str): Path to the input video.

        Returns:
            None
        """
        try:
            # Specify the path to save the video with found detections.
            base_filename = Path(video_path).stem
            video_path_out = os.path.join('test_files', base_filename + '_predictions.mp4')

            # Create a video capture object for the video to predict upon.
            cap = cv2.VideoCapture(video_path)

            # Start reading the video.
            ret, frame = cap.read()

            # Get the dimensions of the video frames.
            H, W, _ = frame.shape

            # Initialize our video writer for saving the output video.
            out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)),
                                  (W, H))

            # Loop through frames read from the video file.
            while ret:
                # Make detections on the current frame
                results = self.model(frame)

                # Write the frame to the output file.
                out.write(frame)

                # Keep reading the frames from the video file until they have all been processed.
                ret, frame = cap.read()

                # Draw detections on the video frames
                frame = self.get_video_detections(results, frame)

            # Release the video capture object and the video writer.
            cap.release()
            out.release()

            # Close all OpenCV windows.
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error detecting over video file: {e}")
            raise

    def detect_over_realtime_feed(self) -> None:
        """
        Process the live video feed for detections.

        Returns:
            None
        """
        try:
            # Open a video capture object for the default camera (index 0)
            self.cap = cv2.VideoCapture(0)

            # Loop while the video capture is open
            while self.cap.isOpened():
                # Read a frame from the video capture
                ret, frame = self.cap.read()

                # Break the loop if there is no more frame to read
                if not ret:
                    break

                # Make detections on the current frame
                results = self.model(frame)

                # Draw detections on the video frames
                frame = self.get_video_detections(results, frame)

                # Display the frame with detections in a window named 'Realtime Stream Detections'
                cv2.imshow('Realtime Stream Detections', frame)

                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Release the video capture object
            self.cap.release()

            # Close all OpenCV windows
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error detecting over realtime feed: {e}")
            raise

    def __del__(self) -> None:
        """
        Releases the video capture when the object is deleted.

        Returns:
            None
        """
        try:
            # Check if the video capture object is not None and is opened.
            if self.cap is not None and self.cap.isOpened():
                # Release the video capture object.
                self.cap.release()
        except Exception as e:
            print(f"Error during object deletion: {e}")
            raise
