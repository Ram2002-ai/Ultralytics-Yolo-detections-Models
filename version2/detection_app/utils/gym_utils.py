import cv2
import numpy as np
from ultralytics import YOLO
import torch

class GymUtils:
    def __init__(self, exercise_type="pushup"):
        self.exercise_type = exercise_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pose_model = YOLO('yolov11n-pose.pt')
        if self.device == 'cuda':
            self.pose_model.to('cuda')
        
        # Rep counting state
        self.rep_count = 0
        self.is_down = False
        self.angle_history = []
        self.state = "up"  # up or down
        
        # Define keypoint indices for COCO pose
        self.SHOULDER_L = 5
        self.SHOULDER_R = 6
        self.ELBOW_L = 7
        self.ELBOW_R = 8
        self.WRIST_L = 9
        self.WRIST_R = 10
        self.HIP_L = 11
        self.HIP_R = 12
        self.KNEE_L = 13
        self.KNEE_R = 14
        self.ANKLE_L = 15
        self.ANKLE_R = 16
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def get_pushup_angle(self, keypoints):
        """Calculate elbow angle for pushup"""
        if keypoints[self.SHOULDER_L] is None or keypoints[self.ELBOW_L] is None or keypoints[self.WRIST_L] is None:
            return None
        
        shoulder = keypoints[self.SHOULDER_L]
        elbow = keypoints[self.ELBOW_L]
        wrist = keypoints[self.WRIST_L]
        
        return self.calculate_angle(shoulder, elbow, wrist)
    
    def get_squat_angle(self, keypoints):
        """Calculate knee angle for squat"""
        if keypoints[self.HIP_L] is None or keypoints[self.KNEE_L] is None or keypoints[self.ANKLE_L] is None:
            return None
        
        hip = keypoints[self.HIP_L]
        knee = keypoints[self.KNEE_L]
        ankle = keypoints[self.ANKLE_L]
        
        return self.calculate_angle(hip, knee, ankle)
    
    def get_bicep_curl_angle(self, keypoints):
        """Calculate elbow angle for bicep curl"""
        if keypoints[self.SHOULDER_L] is None or keypoints[self.ELBOW_L] is None or keypoints[self.WRIST_L] is None:
            return None
        
        shoulder = keypoints[self.SHOULDER_L]
        elbow = keypoints[self.ELBOW_L]
        wrist = keypoints[self.WRIST_L]
        
        return self.calculate_angle(shoulder, elbow, wrist)
    
    def get_overhead_press_angle(self, keypoints):
        """Calculate shoulder angle for overhead press"""
        if keypoints[self.SHOULDER_L] is None or keypoints[self.ELBOW_L] is None or keypoints[self.WRIST_L] is None:
            return None
        
        shoulder = keypoints[self.SHOULDER_L]
        elbow = keypoints[self.ELBOW_L]
        wrist = keypoints[self.WRIST_L]
        
        return self.calculate_angle(shoulder, elbow, wrist)
    
    def update_reps(self, angle, down_threshold=70, up_threshold=160):
        """Update rep count based on angle thresholds"""
        if angle is None:
            return
        
        if angle < down_threshold and self.state == "up":
            self.state = "down"
            self.is_down = True
        elif angle > up_threshold and self.state == "down":
            self.state = "up"
            self.rep_count += 1
            self.is_down = False
    
    def process_frame(self, frame):
        """Process frame for workout monitoring"""
        results = self.pose_model(frame, conf=0.5, verbose=False)
        
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints_data = results[0].keypoints
            annotated = results[0].plot()
            
            # Extract keypoints for the first person
            if keypoints_data.xy is not None and len(keypoints_data.xy) > 0:
                kpts = keypoints_data.xy[0].cpu().numpy()
                confidence = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else None
                
                # Filter valid keypoints (confidence > 0.5)
                keypoints = []
                for i, kpt in enumerate(kpts):
                    if confidence is not None and confidence[i] > 0.5:
                        keypoints.append((int(kpt[0]), int(kpt[1])))
                    else:
                        keypoints.append(None)
                
                # Calculate angle based on exercise type
                angle = None
                if self.exercise_type == "pushup":
                    angle = self.get_pushup_angle(keypoints)
                    down_thresh, up_thresh = 70, 160
                elif self.exercise_type == "squat":
                    angle = self.get_squat_angle(keypoints)
                    down_thresh, up_thresh = 90, 160
                elif self.exercise_type == "bicepcurl":
                    angle = self.get_bicep_curl_angle(keypoints)
                    down_thresh, up_thresh = 60, 150
                elif self.exercise_type == "overheadpress":
                    angle = self.get_overhead_press_angle(keypoints)
                    down_thresh, up_thresh = 90, 160
                else:  # pullup, benchpress use similar elbow logic
                    angle = self.get_pushup_angle(keypoints)
                    down_thresh, up_thresh = 70, 160
                
                # Update rep count if valid angle
                if angle is not None:
                    self.update_reps(angle, down_thresh, up_thresh)
                    
                    # Display angle and rep count
                    cv2.putText(annotated, f"Angle: {int(angle)}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Display feedback
                    if angle < down_thresh + 10:
                        feedback = "Go Up!"
                    elif angle > up_thresh - 10:
                        feedback = "Go Down!"
                    else:
                        feedback = "Good Form!"
                    
                    cv2.putText(annotated, feedback, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display rep count
                cv2.putText(annotated, f"Reps: {self.rep_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Display exercise name
                cv2.putText(annotated, self.exercise_type.upper(), (frame.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return annotated
        else:
            # No pose detected
            annotated = frame.copy()
            cv2.putText(annotated, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated