import cv2
import sys
import dlib

# from FaceRecognitionModel import FaceRecognitionModel

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
 
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)
if __name__ == '__main__' :
    # Read video
    video = cv2.VideoCapture("dance_HD.mp4")

    # fm = FaceRecognitionModel()
    # Find faces in image
    # _, frame = video.read()
    # cv2.imwrite("dance_frame.jpg",frame)
    # face = fm.recognize("dance_frame.jpg")
    # print face
     
    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    print cv2.__version__ 
    tracker = cv2.Tracker_create("MEDIANFLOW") 
    # tracker = cv2.Tracker_create("KCF") 

 
    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
     
    # Define an initial bounding box
    # rectangle(124,95,214,185)
    # dlib.rectangle(124,95,214,185) # this is for out.mp4 whiplash
    # 
    bbox = rect_to_bb(dlib.rectangle(780,130,832,181)) # will
    # bbox = rect_to_bb(dlib.rectangle(509,164,561,216)) # Jimmy

    # rect.top(), rect.right(), rect.bottom(), rect.left()

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)


    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('output_medianflow_will.avi', fourcc, 20.0, (1280,720))# (640,360))
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))

        # Display result
        # cv2.imshow("Tracking", frame)
        out.write(frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()