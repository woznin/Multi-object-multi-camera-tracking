import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "50000"




processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

cap = cv2.VideoCapture("video5.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("results5.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    inputs = processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([frame.shape[:2]])
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes,
        threshold=0.7
    )[0]
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label.item() == 1:  # 1 is the label for humans
            box = [int(i) for i in box.tolist()]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Human: {score:.2f}", 
                        (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()










