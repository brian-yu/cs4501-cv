# cs4501-cv

## Crop board images

`python crop_board.py board_images/IMG_4133.jpg`

## Extract track images (100x100)

`python extract_track_images.py board_images/IMG_4066_deskewed.jpg`

## Labelling track ML training samples

`python label_track_images.py IMG_4066`

## Scoring a board

`python score_board.py board_images/IMG_4066_deskewed.jpg`


## Set up virtual env and all the requirements

python3 -m pip install --user virtualenv

python3 -m venv cvProjTicketToRide  #where you want virtual env

source cvProjTicketToRide/bin/activate

pip install requirements.txt 
