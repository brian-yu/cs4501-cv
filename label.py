import os
import cv2
import sys

"""
Questionable:
    - samples/red-IMG_4066_deskewed-58.jpg (58)
    - samples/red-IMG_4066_deskewed-108.jpg

"""


prefix = sys.argv[1]

sample_dir = 'samples'

files = [
    filename for filename in os.listdir(sample_dir) if filename.startswith(prefix)]

cv2.namedWindow('image')

classes = {"red", "green", "blue", "black", "yellow", "none"}

class_abbrev = {
    'r': 'red',
    'g': 'green',
    'blu': 'blue',
    'bla': 'black',
    'y': 'yellow',
    'n': 'none'
}

for i, filename in enumerate(files):
    print(f"Image {i + 1} out of {len(files)}.")

    img_path = os.path.join(sample_dir, filename)

    print(f"Filename: {img_path}\n")

    img = cv2.imread(img_path)
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF

    label = ""
    while label not in classes:
        label = input("Label: ")
        if label in class_abbrev:
            label = class_abbrev[label]

    plt.close()

    labeled_image_path = os.path.join(sample_dir, f"{label}-{filename}")

    print(f"New filename: {labeled_image_path}\n")

    os.rename(img_path, labeled_image_path)



cv2.destroyAllWindows()

