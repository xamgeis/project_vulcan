import dlib
import cv2 as cv
import openface
from openface import data
import subprocess
import os

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

dirName = raw_input()
alignDirName = "align_face_images/"+dirName+"/"
# dirName = "training-images/"+dirName + "/"
# Create Directory of aligned images
subprocess.call(['mkdir',"align_face_images/"])
subprocess.call(['mkdir',alignDirName])

images = data.iterImgs(dirName) 


alignDlib = openface.AlignDlib("models/dlib/shape_predictor_68_face_landmarks.dat")
# net = openface.TorchNeuralNet(args.networkModel, args.imgDim, cuda=ar) # model used for formatting

# Resize the 10 images from a person
landMarks = [] # a list of (x,y) tuples
imgDim = 96 # we want a 96x96 pixel image
for rgbImg in images:
	img = rgbImg.getRGB()	
	# store img as numpy.ndarray (height, width, 3)
	faceBoxes = alignDlib.getAllFaceBoundingBoxes(img)

	for box in faceBoxes:	
		landMarks = alignDlib.findLandmarks(img,box)
		# transform and align faces
		face_align = alignDlib.align(imgDim=imgDim,rgbImg=img,landmarks=landMarks) # returns numpy.ndarry of the aligned RGB image
		if face_align is None:
			raise Exception("Unable to align image: {}".format(imgPath))
    
		# store image
		cv.imwrite(alignDirName+rgbImg.name+".jpg",face_align)
./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96
# subprocess.call(['./util/align-dlib.py','./training-images/','align','outerEyesAndNose','./aligned-images/','--size',"96"])
# ./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/
subprocess.call(['./batch-represent/main.lua',"-outDir","./generated-embeddings/","-data","./align_face_images/"])

# ./demos/classifier.py train ./generated-embeddings/
subprocess.call(["./demos/classifier.py","train","./generated-embeddings/"])
# embeddings generated in labels.csv and reps.csv


# # Convert to Vectorize form
# rep1 = net.forward(alignedFace)
# rep2 = net.forward(alignedFace)
