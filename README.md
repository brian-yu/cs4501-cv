# ResNet to Ride: Automatic scoring of the Ticket to Ride board game

# Score a board

```
python3 -m pip install --user virtualenv
python3 -m venv cvProjTicketToRide  #where you want virtual env
source cvProjTicketToRide/bin/activate
pip install -r requirements.txt 
```

1. `python score_board.py board_images/IMG_4066.jpg`
2. Select corners
3. Press 'n'!

# Dataset collection utilities

## Crop board images

`python crop_board.py -i board_images/IMG_4133.jpg`

## Extract track images (100x100)

`python extract_track_images.py board_images/IMG_4066_deskewed.jpg`

## Labelling track ML training samples

`python label_track_images.py IMG_4066`

## Edge detection with pretrained Caffe model 

`cd edge_detection`
`python edge.py --input ../board_images/IMG_4173.jpg`
