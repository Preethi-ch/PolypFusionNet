import time
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='../runs/train/train_yolov5n/weights/best.pt', device='cpu')

start = time.time()
model('../test_image.jpg')
end = time.time()

print("Inference Time:", (end-start)*1000, "ms")
print("FPS:", 1/(end-start))
