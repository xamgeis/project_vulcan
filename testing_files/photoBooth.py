import cv2 as cv
import subprocess
from time import sleep

def startCamera(resolutionX, resolutionY, vFlip=False,hFlip=False):
	cam = cv.VideoCapture(0)
	return cam

def takePicture(camera):
	print("Taking picture...")
	ret, frame = camera.read()
	return frame
	
def storeImage(img,dirName,photoName):
	# store into directory
	cv.imwrite(dirName+str(photoName)+".jpg",img)

def startPreview():
	print("Displaying preview...")

def stopPreview():
	print("Stopping preview...")


# ##################################################################
#
# Main Program
#
# ##################################################################
# Take 10 pictures of a person in a photobooth style

# Start Camera
cam = startCamera(100,100)

# startPreview()
print "Enter Name: "
dirName = raw_input()
dirName = "face_images/"+dirName+"/"
subprocess.call(['mkdir',dirName])

nextWaitTime = 1
# Initiate countdown
totalPhotos = 10
for photoNum in range(totalPhotos):
	sleep(nextWaitTime)
	img = takePicture(cam);
	# cv.imshow("Image",img)
	storeImage(img,dirName,photoNum)
	
# stopPreview()
cam.release()
cv.destroyAllWindows()