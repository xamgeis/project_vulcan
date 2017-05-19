import cv2
import sys
import dlib

from FaceRecognitionModel import FaceRecognitionModel

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
    video = cv2.VideoCapture("videos/dance_HD.mp4")

    fm = FaceRecognitionModel()
    # Find faces in image
    _, frame = video.read()
    cv2.imwrite("frame.jpg",frame)
    faces = fm.recognize("frame.jpg")

    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    print cv2.__version__
    #tracker = cv2.Tracker_create("MEDIANFLOW")
    tracker = cv2.Tracker_create("KCF")


    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    # rectangle(124,95,214,185)
    if faces != []:
		#create an array of bounding boxes to track
		i = 0;
		for face in faces:
			bbox[i] = rect_to_bb(face[3]) #set bbox at this index to be the bounding box
        # Initialize tracker with first frame and bounding box
			ok = tracker.init(frame, bbox[i])
			i = i+1
        isTracking = True
    else:
        bbox = null
        isTracking = False

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)



    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('./video/output_medianflow_will.avi', fourcc, 20.0, (1280,720))# (640,360))    
    frameCount = 0
    n = 20
    while True:
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break

        # determine if there is a face
        frameCount = frameCount % n
        if frameCount == 0:
            cv2.imwrite("./video/face.jpg", frame)
            faces = fm.recognize("./video/face.jpg")
            if faces != []:
                tracker.clear()
                #tracker.update(frame,rect_to_bb(faces[0][3]))
				for face in faces:
					tracker.init(frame,rect_to_bb(face[3]))
        frameCount += 1

        # Update tracker
		for box in bbox:
			ok, box = tracker.update(frame)

        # Draw bounding box
			if ok:
				p1 = (int(box[0]), int(box[1]))
				p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
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
