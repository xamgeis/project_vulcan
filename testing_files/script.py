import numpy as np
import cv2 as cv
import time
import faceAPI
from faceAPI.FaceRecognitionModel import FaceRecognitionModel

debug = True
onComputer = True

fm = FaceRecognitionModel()
# fm.learnAll("./training-images/")
ok, frame = video.read("jo.mp4")
cv.imwrite("./test-images/jo.jpg",frame)
fm.recognize("./test-images/jo.jpg")
# fm.recognize("./test-images/will_1.jpg")
# fm.recognize("./test-images/arnold_1.jpeg")

#Set dimensions of frame as 320x240 for better performance on Pi
# FRAME_WIDTH = 320
# FRAME_HEIGHT = 240
# cap = cv.VideoCapture(0)
# cap.set(cv.cv.CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# cap.set(cv.cv.CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# if not onComputer:
#     camera = CamControl()
#     camera.up(50, 9)
    
# time.sleep(1)
# while True:

#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     cv.flip(frame, 1, frame)  # flip the image

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release capture
# cap.release()
# cv.destroyAllWindows()
