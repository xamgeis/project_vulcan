
import openface
import os

from Classifier import Classifier 
from Processor import Processor
from Util import Util as util
class FaceRecognitionModel():
	

	# File dirs
	fileDir = os.path.dirname(os.path.realpath(__file__))
	modelDir = os.path.join(fileDir, 'models')
	dlibModelDir = os.path.join(modelDir, 'dlib')
	openfaceModelDir = os.path.join(modelDir, 'openface')
	networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
	embeddingsDir = os.path.join(fileDir, 'generated-embeddings')
	classifierModelDir = os.path.join(embeddingsDir,'classifier.pkl')

	# List of people
	
	def __init__(self):
		self.SVM = Classifier(verbose=True)
		self.processor = Processor(verbose=True)
		self.util = util()

	def learn(self):
		"""
		Learns a new face.
		"""
		return ""

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
			self.processor.processImage(imgObject,isTrain=True)
		
		# Train face detection model 
		self.SVM.train() 
		# This will generate a new file called ./generated-embeddings/classifier.pkl. 
		# This file has the SVM model you'll use to recognize new face
		print("Finished learning all knowledge!")

if __name__ == '__main__':
	fm = FaceRecognitionModel()
	# fm.learnAll("./training-images/")
	fm.recognize("./test-images/max_1.jpg")
	fm.recognize("./test-images/will_1.jpg")
	fm.recognize("./test-images/arnold_1.jpeg")
