import torch
from models.common import DetectMultiBackend
from utils.general import (check_img_size, cv2,
                           non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
import numpy as np
import time
import io
import base64
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

data = 'data/data.yaml'  # Configuração do arquivo de dados

app = FastAPI()

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def detect(im, model, device, iou_threshold=0.45, confidence_threshold=0.25):
    imgsz = (640, 640)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)
    imgs = im.copy()

    image, _, _ = letterbox(im, auto=False)
    image = image.transpose((2, 0, 1))
    img = torch.from_numpy(image).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    start = time.time()
    pred = model(img, augment=False)
    fps = 1 / (time.time() - start)
    pred = non_max_suppression(pred, confidence_threshold, iou_threshold, None, False, max_det=10)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], imgs.shape).round()
            annotator = Annotator(imgs, line_width=3)
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
    return imgs, fps

@app.post('/api/predict-file')
async def predict_file(
    file: UploadFile,
    model_link: str = Form(...),
    iou_threshold: float = Form(0.45),
    confidence_threshold: float = Form(0.25)
):
    try:
        print(f"Arquivo recebido: {file.filename}")
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(pil_img)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Dispositivo: {device}")

        model_path = f'weights/best.pt'
        print(f"Carregando modelo: {model_path}")
        model = DetectMultiBackend(model_path, device=device, dnn=False, data=data, fp16=False)

        processed_img, fps = detect(img_array, model, device, iou_threshold, confidence_threshold)
        # Corrigir cores de BGR para RGB antes de codificar
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        success, encoded_img = cv2.imencode(".jpg", processed_img_rgb)
        if not success:
            raise HTTPException(status_code=500, detail="Erro ao codificar imagem")

        encoded_base64 = base64.b64encode(encoded_img).decode('utf-8')
        print(f"Processamento concluído. FPS: {fps}")
        return {"data": encoded_base64, "fps": fps}

    except Exception as e:
        print(f"Erro interno: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)