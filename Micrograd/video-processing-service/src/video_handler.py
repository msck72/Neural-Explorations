import cv2

from utils import get_fourcc_from_suffix

SAMPLING_FREQ = 1

class VideoHandler:
    def __init__(self, video=None):
        if(video != None):
            self.video = video

    def set_video(self, video):
        self.video = video
    
    def detect_objects(self, model_handler, processed_video_path, suffix):
        if(self.video is None):
            print("Video is not set")
            return

        print('suffix = ', suffix)
        cap = cv2.VideoCapture(self.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('fps = ', fps)
        writer = cv2.VideoWriter(
            processed_video_path,
            cv2.VideoWriter_fourcc(*get_fourcc_from_suffix(suffix)),
            fps / SAMPLING_FREQ,
            (width, height)
        )

        frame_idx = 0
        tracking_details = {}

        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            # writer.write(frame)
            if frame_idx % SAMPLING_FREQ == 0:
                frame = model_handler.detect_objects(frame, frame_idx, tracking_details)
                writer.write(frame)

            frame_idx += 1
        
        cap.release()
        writer.release()

        return tracking_details

    def blur_video(self, blur_frames, processed_video_path, suffix):
        if(self.video is None):
            print("Video is not set")
            return

        print('suffix = ', suffix)
        cap = cv2.VideoCapture(self.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            processed_video_path,
            cv2.VideoWriter_fourcc(*get_fourcc_from_suffix(suffix)),
            fps / SAMPLING_FREQ,
            (width, height)
        )

        frame_idx = 0

        ptr_bf = 0
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            if frame_idx % SAMPLING_FREQ == 0:
                while (ptr_bf < len(blur_frames) and frame_idx == blur_frames[ptr_bf]['frame_id']):
                    print('blurring frame ', frame_idx)
                    x1, y1, x2, y2 = map(int, blur_frames[ptr_bf]['bbox'])

                    roi = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
                    ptr_bf += 1
                
                writer.write(frame)

            frame_idx += 1
        
        cap.release()
        writer.release()

        return
