import torch
from unet import UNet4k
import os
from utils.data_loading import load_image
from torch.utils.data import DataLoader
import numpy as np
from vi3o import view, viewsc
from vi3o.image import imwrite
from torchvision.transforms.functional import to_tensor
from evaluate import map2points
import cv2
from torch import sigmoid_, softmax
import os

# pil_img = load_image("pdata/eval/imgs/bf414f6f-8c80-4596-9f9d-27fc9e35fd1c.jpg")
pil_img = load_image("../python-monolith/dline.jpg")

if True:
    model = UNet4k(3, 1, 1)
    checkpoint = "checkpoints4/checkpoint_epoch500.pth"
    ofn = "v4.jpg"
else:
    model = UNet4k(3, 3, 1)
    checkpoint = "checkpoints2/checkpoint_epoch50.pth"
    ofn = "v2.jpg"


device = torch.device('cuda')
model = model.to(device=device, memory_format=torch.channels_last)

state_dict = torch.load(checkpoint, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
model.load_state_dict(state_dict)
model.eval()

image = to_tensor(pil_img)[None].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

with torch.inference_mode():
    mask_pred, endpoint_pred = model(image)
    if model.n_classes == 1:
        mask_pred = sigmoid_(mask_pred)
    else:
        mask_pred = softmax(mask_pred, 1)

pkts = map2points(endpoint_pred[0], 0.6)
drw = np.array(pil_img)
for x, y, _ in pkts:
    cv2.circle(drw, (int(x), int(y)), 10, (255,0,0), -1)
if mask_pred.shape[1] == 3:
    drw[:,:,1] = 255 * mask_pred[0, 1].cpu().numpy()
    drw[:,:,2] = 255 * mask_pred[0, 2].cpu().numpy()
else:
    drw[:,:,2] = 255 * mask_pred[0, 0].cpu().numpy()
# view(drw)
# view(255 * mask_pred[0, 2].cpu().numpy())

imwrite(drw, ofn)
os.system(f"xzgv {ofn}")