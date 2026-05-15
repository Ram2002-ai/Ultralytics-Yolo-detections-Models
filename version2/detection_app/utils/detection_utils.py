import cv2
import numpy as np
from ultralytics import YOLO
import torch

class DetectionUtils:
    def __init__(self):
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models once during initialization
        self.det_model = YOLO('yolov8n.pt')  # Object detection
        self.seg_model = YOLO('yolov8n-seg.pt')  # Segmentation
        self.pose_model = YOLO('yolov8n-pose.pt')  # Pose estimation
        
        # Move models to GPU if available
        if self.device == 'cuda':
            self.det_model.to('cuda')
            self.seg_model.to('cuda')
            self.pose_model.to('cuda')
    
    def detect_objects(self, frame, conf_threshold=0.45):
        """Standard object detection with bounding boxes and labels"""
        results = self.det_model(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        return results, annotated
    
    def count_objects(self, frame, conf_threshold=0.45):
        """Object detection with class-wise counting"""
        results = self.det_model(frame, conf=conf_threshold, verbose=False)
        
        # Count objects per class
        if results[0].boxes is not None:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            names = results[0].names
            
            counts = {}
            for class_id in class_ids:
                class_name = names[class_id]
                counts[class_name] = counts.get(class_name, 0) + 1
            
            # Annotate frame with counts
            annotated = results[0].plot()
            y_offset = 30
            for class_name, count in counts.items():
                cv2.putText(annotated, f"{class_name}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
        else:
            annotated = frame
        
        return results, annotated
    
    def segment_objects(self, frame, conf_threshold=0.45):
        """Instance segmentation with masks"""
        results = self.seg_model(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        return annotated
    
    def pose_estimate(self, frame, conf_threshold=0.45):
        """Pose estimation with keypoints and skeleton"""
        results = self.pose_model(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        return annotated
    
    def customer_detection(self, frame, conf_threshold=0.45):
        """Person detection and tracking (customer counting)"""
        results = self.det_model(frame, conf=conf_threshold, verbose=False)
        
        # Filter only persons (class 0 in COCO)
        annotated = frame.copy()
        if results[0].boxes is not None:
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            person_count = 0
            for i, class_id in enumerate(class_ids):
                if class_id == 0:  # person class
                    person_count += 1
                    box = boxes.xyxy[i].cpu().numpy().astype(int)
                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Customer {person_count}", (box[0], box[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display total count
            cv2.putText(annotated, f"Total Customers: {person_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated