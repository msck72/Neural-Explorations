import base64

import cv2
from ultralytics import YOLO

class ModelHandler:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
    
    def detect_objects(self, img, frame_id, tracking_details):
        result = self.model.track(
            img,
            persist=True,
            tracker="botsort.yaml",
            verbose=False
        )
        result = result[0].boxes
        for box in result:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            label = f"{self.model.names[cls]} {conf:.2f}"

            box_id = int(box.id.item())
            if box_id not in tracking_details:
                crop = img[y1:y2, x1:x2].copy()

                success, buffer = cv2.imencode(".png", crop)
                if success:
                    crop_b64 = base64.b64encode(buffer).decode("utf-8")
                else:
                    print('img cannot be converted to png')
                tracking_details[box_id] = {
                    "cls": cls,
                    "crop": crop_b64,
                    "bboxes":[]
                }
            
            tracking_details[box_id]['bboxes'].append({
                "frame_id": frame_id,
                "conf": conf,
                "bbox": (x1, y1, x2, y2)
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return img

    def reset_tracking(self):
        if hasattr(self.model.predictor, "trackers"):
            for tracker in self.model.predictor.trackers:
                tracker.reset()