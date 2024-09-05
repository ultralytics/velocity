# Ultralytics YOLO 🚀, AGPL-3.0 License https://ultralytics.com/license

import os

import cv2

import utils.common

PATH = "/Users/glennjocher/Downloads/DATA/VSM/chessboard/IMG_4414.MOV"
path, file, _, _ = utils.common.filenamesplit(PATH)
newdir = path + file + "/"

if not os.path.exists(newdir):
    os.mkdir(newdir)

cap = cv2.VideoCapture(PATH)
for i in range(0, 2000, 10):
    cap.set(1, i)
    success, im = cap.read()  # read frame
    if success:
        print(f"image {cap.get(cv2.CAP_PROP_POS_FRAMES):g}/{cap.get(cv2.CAP_PROP_FRAME_COUNT):g} ...")
        cv2.imwrite(newdir + str(i) + ".jpg", im)
    else:
        cap.release()
        break
print("Done.")
