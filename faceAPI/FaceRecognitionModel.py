
import openface
import os
import subprocess 
from Classifier import Classifier 
from Processor import Processor
from Util import Util as util

import cv2 as cv
class FaceRecognitionModel():
		
	# List of people
	
	def __init__(self):
		self.SVM = Classifier(verbose=True)
		self.processor = Processor(verbose=True)
		self.util = util()

	def learn(self,dir):
		"""
		Learns a new face from a set of images in a directory.
		"""
		imgs = openface.data.iterImgs(dir)
		for imgObject in imgs:
			self.processor.processImage(imgObject,isTrain=True)
		
		# Train face detection model 
		self.SVM.train() 

	def recognize(self,img):
		"""
		Finds faces in an image and labels if they are recognized or not.
		"""
		
		# Create image object from img
		imgName = os.path.basename(img)
		imgObject = openface.data.Image("?",imgName,img)
		# Get face rAepresentation: list of (bb, face vectors)
		reps = self.processor.processImage(imgObject)

		# Pass reps into SVM classifier to get predictions
		faces = self.SVM.infer(self.util.classifierModelDir,reps) # (classifierModelDir, imgs,multiple=False,verbose=False)
		print faces
		# list of (person's name, confidence , bounding box)
		# list of (string, float , rectangle object)
		return faces

	def find(self):
			"""
			Find all faces in the image and return locations (bounding boxes)
			"""
			return ""

	def learnAll(self,dir):
		"""
		Learns from the entire dataset using the specified training-images directory.
		"""
		# process all images in the directory
		imgs = openface.data.iterImgs(dir)
		for imgObject in imgs:
			reps = self.processor.processImage(imgObject,isTrain=True)

		# Create a directory for generated embeddings (if not already created)
		openface.helper.mkdirP(self.util.embeddingsDir)
		# create CSV files using csvigo called from openface source code
		subprocess.call(['./batch-represent/main.lua',"-outDir",self.util.embeddingsDir ,"-data", self.util.alignedImgsDir])
		# TODO change string directory ^ for batch-represent to a define package

		# Train face detection model 
		self.SVM.train() 
		# This will generate a new file called ./generated-embeddings/classifier.pkl. 
		# This file has the SVM model you'll use to recognize new face
		print("Finished learning all knowledge!")

# if __name__ == '__main__':
# 	fm = FaceRecognitionModel()
#  	#fm.learnAll("./small-training-images/")



# 	imgs = openface.data.iterImgs("./testing-images");
# 	for imgObject in imgs:
# 		face = fm.recognize(imgObject.path);
# 		img = fm.processor.markFace(imgObject.getBGR(),face)
# 		text = "./marked-testing-images/"+"marked-"+ imgObject.name + ".png"
# 		cv.imwrite(text,img)
