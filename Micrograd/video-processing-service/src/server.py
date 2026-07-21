import os
from pathlib import Path
import shutil
import tempfile
import json

from fastapi import Body, FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

from model_handler import ModelHandler
from video_handler import VideoHandler

OUTPUT_DIR = Path("processed_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

model_handler = None
video_handler = None
request_id_to_tracking = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_handler, video_handler
    model_handler = ModelHandler()
    video_handler = VideoHandler()
    print("Application started")

    yield

    print("Cleaning up...")

app = FastAPI(title="Image Processing API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image: str


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/process_video")
async def process_video(video: UploadFile = File(...), request_id: str = Form(...)):
    suffix = os.path.splitext(video.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        shutil.copyfileobj(video.file, temp)
        input_path = temp.name
    
    saved_path = OUTPUT_DIR / f"request_{video.filename}"
    shutil.copy2(input_path, saved_path)

    output_fd, output_path = tempfile.mkstemp(suffix=suffix)
    os.close(output_fd)
    
    video_handler.set_video(input_path)
    request_id_to_tracking[request_id] = video_handler.detect_objects(model_handler, output_path, suffix)
    model_handler.reset_tracking()

    # saved_path = OUTPUT_DIR / f"processed_{video.filename}"
    # shutil.copy2(output_path, saved_path)

    os.remove(input_path)

    return FileResponse(
        output_path,
        media_type=f"video/{suffix}",
        filename=f"processed.{suffix}",
    )

@app.post("/tracking_details")
async def get_tracking_details(request_id: str = Form(...)):
    response = request_id_to_tracking[request_id]
    request_id_to_tracking.pop(request_id, None)
    return response

@app.post("/blur_video")
async def blur_video(video: UploadFile = File(...), blur_info: str = Form(...)):
    suffix = os.path.splitext(video.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        shutil.copyfileobj(video.file, temp)
        input_path = temp.name

    output_fd, output_path = tempfile.mkstemp(suffix=suffix)
    os.close(output_fd)
    
    video_handler.set_video(input_path)
    # for k, v in request:
    # print(blur_info) 
    blur_info = json.loads(blur_info)
    # print(type(blur_info))
    blur_frames = []
    for _, v in blur_info.items():
        blur_frames.extend(
            v['bboxes']
        )
    sorted_blur_frames = sorted(blur_frames, key=lambda x: x['frame_id'])
    # for i in sorted_blur_frames:
    #     print(i['frame_id'])

    video_handler.blur_video(sorted_blur_frames, output_path, suffix)

    # saved_path = OUTPUT_DIR / f"processed_{video.filename}"
    # shutil.copy2(output_path, saved_path)

    return FileResponse(
        output_path,
        media_type=f"video/{suffix}",
        filename=f"processed.{suffix}",
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9877,
        reload=False,
    )