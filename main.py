import json
import threading
import time
import signal
import sys

import cv2
from imutils import resize

import daisykit
from daisykit.utils import get_asset_file

from playsound import playsound


# 183-16-369
def exit_all(sig, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, exit_all)


config = {
    "face_detection_model": {
        "model": get_asset_file("models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.param"),
        "weights": get_asset_file("models/face_detection/yolo_fastest_with_mask/yolo-fastest-opt.bin"),
        "input_width": 320,
        "input_height": 320,
        "score_threshold": 0.7,
        "iou_threshold": 0.5,
        "use_gpu": False
    },
    "with_landmark": False
}
face_detector_flow = daisykit.FaceDetectorFlow(json.dumps(config))

vid = cv2.VideoCapture(0)

cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

last_warning_sound = 0 # To save time when last warning sound played
logo = cv2.imread("logo.png")
logo = resize(logo, width=1200)
logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)

while(True):
    ret, frame = vid.read()

    frame = resize(frame, width=1200)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector_flow.Process(frame)
    face_detector_flow.DrawResult(frame, faces)

    for face in faces:
        if face.wearing_mask_prob < 0.5: # Probability of wearing a mask < 0.5 => No mask

            if face.w < 50 or face.h < 50:
                continue

            if time.time() - last_warning_sound > 6.0:
                threading.Thread(target=playsound, args=("warning.mp3",), daemon=True).start()
                last_warning_sound = time.time() # Assign current time to last_warning_sound


    if time.time() - last_warning_sound < 6.0: # Show "Warning !!" notification
        cv2.putText(frame, 'Warning!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
             1.0, (255, 0, 0), 2, cv2.LINE_AA)
        # Red border
        frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, None, value = (255, 0, 0))
        # Resize frame width to 1200
        frame = resize(frame, width=1200)




    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', frame)
    c = cv2.waitKey(1)
    if c == ord('q'): # Exit when user press "q"
        exit_all(0, 0)
