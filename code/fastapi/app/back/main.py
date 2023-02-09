from pathlib import Path
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.param_functions import Depends
from pydantic import Json
from pdf2image import convert_from_bytes
from pycocotools.coco import COCO
from mmdet.apis import init_detector
import numpy as np
import sys

sys.path.append("/opt/ml/input/code/fastapi/app/back")
from inference import *
from utils import *
from recognition import *
from database import *
import os
from datetime import datetime

# Settings
model_config = "/opt/ml/input/data/models/100/detection_config.py"
model_weight = "/opt/ml/input/data/models/100/best_bbox_mAP_epoch_26.pth"

# Back-end App
app = FastAPI()

# Load Model (Detection, OCR)
detector = init_detector(model_config, model_weight, device="cuda:0")
ocr_model = load_ocr_model(
    save_model="/opt/ml/input/data/models/recog_model.pth", device="cuda:0"
)


# Load Ann, Image, Predict
@app.post("/predict/{exam_info}")
def predict(exam_info: str, file: UploadFile = File(...)):
    # get annotation info
    answer, q_bbox, img_shape = get_info_from_db(exam_info)

    # pdf -> np.array
    images = convert_from_bytes(file.file._file.read(), dpi=100)
    images_np = [np.array(image) for image in images]

    # make directory for log
    infer_time = str(datetime.now()).replace(" ", "_")
    if not os.path.isdir(f"/opt/ml/input/code/fastapi/app/log/{infer_time}"):
        os.mkdir(f"/opt/ml/input/code/fastapi/app/log/{infer_time}")
    for idx in range(len(images_np)):
        Image.fromarray(images_np[idx]).save(
            f"/opt/ml/input/code/fastapi/app/log/{infer_time}/{idx}_original.jpg",
            "JPEG",
        )
    for idx in range(len(images_np)):
        Image.fromarray(images_np[idx]).save(
            f"/opt/ml/input/code/fastapi/app/log/{infer_time}/{idx}_predict.jpg",
            "JPEG",
        )

    # Model Inference
    inference = Inference_v2(
        images=images_np,
        detector=detector,
        q_bbox=q_bbox,
        answer=answer,
        img_shape=img_shape,
        time=infer_time,
        ocr_model=ocr_model,
    )
    scoring_img, log_pred = inference.main()

    # Insert log to DB
    insert_log(log_pred, exam_info, infer_time)

    # make scored image, return to front-end
    imgByteArr = io.BytesIO()
    scoring_img[0].save(
        imgByteArr, save_all=True, append_images=scoring_img[1:], format="PDF"
    )

    return Response(imgByteArr.getvalue(), media_type="application/pdf")
