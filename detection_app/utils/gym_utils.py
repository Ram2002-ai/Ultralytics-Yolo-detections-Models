import cv2
from ultralytics import solutions

class GymUtils:
    def __init__(self, model_path="yolov11n-pose.pt", exercise_type="pushup"):
        self.exercise_type = exercise_type
        # keypoint indices for different exercises
        self.keypoints_map = {
            "pushup": [5, 7, 9],
            "squat": [5, 7, 9, 11, 13],
            "pullup": [5, 7, 9],
            "benchpress": [5, 7, 9],
            "bicepcurl": [5, 7, 9],
            "overheadpress": [5, 7, 9]
        }
        kpts = self.keypoints_map.get(exercise_type, [5, 7, 9])
        self.gym = solutions.AIGym(
            show=False,
            kpts=kpts,
            model=model_path,
            line_width=4,
            verbose=False
        )

    def process_frame(self, frame):
        results = self.gym(frame)
        # The annotated image is stored in results.im0 (works in latest ultralytics)
        if hasattr(results, 'im0'):
            return results.im0
        elif hasattr(results, 'plot'):
            return results.plot()
        else:
            # fallback: return original frame
            return frame

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed = self.process_frame(frame)
            out.write(processed)
        cap.release()
        out.release()
        return output_path