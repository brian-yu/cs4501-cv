import cv2
import numpy as np
import pickle
import sys
import os
from collections import Counter, defaultdict
import time
import json

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image


tracks = pickle.load( open( "tracks.p", "rb" ) )

RATIO = 79.5 / 52.5
WIDTH = 1500
HEIGHT = int(WIDTH / RATIO) # 990 if width = 1500

TRAIN_WIDTH = 100
TRAIN_HEIGHT = 100

# EXPECT CROPPED + DESKEWED IMAGE FOR NOW

img_path = sys.argv[1]
path_split = os.path.split(img_path)
filename = path_split[-1]
img_dir = os.path.join(*path_split[:-1])
image = cv2.imread(img_path)

prefix = filename.split(".")[0]


def deskew(image, corner_pts):
    pts_src = np.array(corner_pts)

    pts_dst = np.array([[0, 0], [WIDTH-1, 0], [WIDTH-1, HEIGHT-1], [0, HEIGHT-1]])

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(image, h, (WIDTH, HEIGHT))

    return im_out

def predict(image, model):

    label_names = ["red", "green", "blue", "black", "yellow", "none"]

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_img = preprocess(img).unsqueeze(0)

    output = model(input_img)

    pred = torch.argmax(output[0]).item()

    return label_names[pred]

def extract(image, pts, target_width, target_height):

    pts_src = np.array(pts)

    pts_dst = np.array(
        [[0, 0], [target_width-1, 0],
        [target_width-1, target_height-1], [0, target_height-1]])
    # top left, top right, bottom right, bottom left

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_out = cv2.warpPerspective(image, h, (target_width, target_height))

    return im_out


corner_pts = json.loads(open('corner_samples/corner_labels.json', 'r').read())[filename]

image = deskew(image, corner_pts)

print(f"{len(tracks)} tracks")

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)
model.load_state_dict(torch.load("ticket_to_ride_model.pt", map_location='cpu'))
model.eval()

labeled_boxes = []
score_map = {
    1: 1,
    2: 2,
    3: 4,
    4: 7,
    5: 10,
    6: 15
}
color_scores = defaultdict(int)

print("Scoring...")
start = time.time()
for track in tracks:
    colors = Counter()
    for box in track:
        pts = np.array(box, np.int32).reshape((-1,1,2))

        box_img = extract(image, box, TRAIN_WIDTH, TRAIN_HEIGHT)

        pred_color = predict(box_img, model)
        labeled_boxes.append((pts, pred_color))
        colors[pred_color] += 1

    color = colors.most_common(1)[0][0]

    if color != 'none':
        color_scores[color] += score_map[len(track)]

print(f"Finished in {time.time() - start} seconds.")

print(color_scores)

colors = {
    "red": (0,0,255),
    "green": (0,255,0),
    "blue": (255,0,0),
    "black": (0,0,0),
    "yellow": (0,255,255),
    "none": (255,255,255)
}

for pts, color in labeled_boxes:
    cv2.polylines(image,[pts],True, colors[color], 3)

cv2.imshow('board', image)
cv2.waitKey(0)
cv2.destroyAllWindows()