import cv2
import numpy as np
from ultralytics import YOLO
import random

class DetectionUtils:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        # COCO class names (for object detection)
        self.class_list = self._get_coco_classes()
        self.detection_colors = self._generate_colors()

    def _get_coco_classes(self):
        # Standard COCO class names (80 classes)
        coco_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        return coco_classes

    def _generate_colors(self):
        colors = []
        for _ in range(len(self.class_list)):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            colors.append((b, g, r))
        return colors

    def detect_objects(self, frame, conf_threshold=0.45):
        results = self.model.predict(source=[frame], conf=conf_threshold, verbose=False)
        detections = []
        if len(results[0].boxes) > 0:
            boxes_data = results[0].boxes
            for i in range(len(boxes_data)):
                box = boxes_data[i]
                cls_id = int(box.cls.numpy()[0])
                conf = float(box.conf.numpy()[0])
                bb = box.xyxy.numpy()[0]
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_list[cls_id],
                    "confidence": conf,
                    "bbox": bb
                })
        return detections, results[0].plot() if detections else frame

    def count_objects(self, frame, conf_threshold=0.45):
        detections, annotated_frame = self.detect_objects(frame, conf_threshold)
        count = len(detections)
        cv2.putText(annotated_frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return count, annotated_frame

    def segment_objects(self, frame, conf_threshold=0.45):
        # Segmentation model
        seg_model = YOLO("yolov8n-seg.pt")
        results = seg_model.predict(source=[frame], conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot() if results else frame
        return annotated_frame

    def pose_estimate(self, frame, conf_threshold=0.45):
        # Pose model
        pose_model = YOLO("yolov8n-pose.pt")
        results = pose_model.predict(source=[frame], conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot() if results else frame
        return annotated_frame

    def customer_detection(self, frame, conf_threshold=0.45):
        # Filter only "person" class (class_id = 0)
        detections, _ = self.detect_objects(frame, conf_threshold)
        person_detections = [d for d in detections if d["class_name"] == "person"]
        # Draw custom annotations
        annotated_frame = frame.copy()
        for det in person_detections:
            bb = det["bbox"]
            cv2.rectangle(annotated_frame, (int(bb[0]), int(bb[1])),
                          (int(bb[2]), int(bb[3])), (0, 255, 0), 3)
            label = f"Customer {det['confidence']:.2f}"
            cv2.putText(annotated_frame, label, (int(bb[0]), int(bb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Customers: {len(person_detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated_frame