"""
Author: Jacob Pitsenberger
Program: utils.py
Version: 1.0
Project: Detecting Filtered Classes with YOLOv8 Pretrained Model
Date: 12/5/2023
Purpose: This module conains ...
Uses: N/A
"""

import numpy as np


def create_coco_classes_dict() -> dict:
    """
    Create and return a dictionary mapping class IDs to class names for the COCO dataset.

    Returns:
        dict: Dictionary mapping class IDs to class names.
    """
    coco_classes = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush',
    }
    return coco_classes


def non_max_suppression(boxes: np.ndarray, confidences: np.ndarray, threshold: float = 0.5) -> list:
    """
    Adapted from: https://github.com/computervisioneng/automatic-number-plate-recognition-python/blob/master/yolov3-from-opencv-object-detection/util.py

    Apply Non-Maximum Suppression (NMS) to filter out duplicate detections.

    Args:
        boxes (np.ndarray): A numpy array containing bounding boxes in [x1, y1, x2, y2] format.
        confidences (np.ndarray): A numpy array containing confidence scores for each bounding box.
        threshold (float): The IoU (Intersection over Union) threshold above which duplicate detections will be removed.

    Returns:
        list: A list containing the indices of boxes to keep after NMS.
    """
    # Convert boxes to [x1, y1, x2, y2] format for NMS
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate the area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort boxes by their confidence scores in descending order
    order = confidences.argsort()[::-1]

    indices_to_keep = []  # List to store the indices of boxes to keep after NMS

    while order.size > 0:
        # Keep the box with the highest confidence score
        best_box_idx = order[0]
        indices_to_keep.append(best_box_idx)

        # Calculate IoU with the other boxes
        xx1 = np.maximum(x1[best_box_idx], x1[order[1:]])
        yy1 = np.maximum(y1[best_box_idx], y1[order[1:]])
        xx2 = np.minimum(x2[best_box_idx], x2[order[1:]])
        yy2 = np.minimum(y2[best_box_idx], y2[order[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)

        intersection = width * height

        # Calculate IoU
        iou = intersection / (area[best_box_idx] + area[order[1:]] - intersection)

        # Remove boxes with IoU greater than the threshold
        overlapping_boxes = np.where(iou > threshold)[0]
        order = order[overlapping_boxes + 1]

    return indices_to_keep
