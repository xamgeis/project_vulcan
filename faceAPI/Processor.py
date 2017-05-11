import cv2 as cv
import time
import os
import openface

from Util import Util

class Processor:
	dirs = Util()

	def __init__(self,usingCuda=False,verbose=False):
		self.imgDim = 96
		networkModel = os.path.join(Processor.dirs.openfaceModelDir,'nn4.small2.v1.t7')
		self.net = openface.TorchNeuralNet(networkModel, self.imgDim, cuda=usingCuda) # model used for formatting	
		self.dlibFacePredictor = os.path.join(Processor.dirs.dlibModelDir, "shape_predictor_68_face_landmarks.dat")
		self.align = openface.AlignDlib(self.dlibFacePredictor)
		self.verbose = verbose
		openface.helper.mkdirP(Processor.dirs.alignedImgsDir)
	

	def processImage(self,imgObject,isTrain=False):
		"""
		Get aligned reps and their bounding boxes
		"""
		reps = [] 
		print("=== {} ===".format(imgObject.path))

		# Preprocess image
		if isTrain: 
			outDir = os.path.join(Processor.dirs.alignedImgsDir, imgObject.cls)
			openface.helper.mkdirP(outDir)
			outputPrefix = os.path.join(outDir, imgObject.name)
			imgName = outputPrefix + ".png"
			# TODO check if file is already found. if so, then nothing needs to be done.
			# Otherwise, continue.
			if os.path.isfile(imgName):
				if self.verbose:
					print("  + Already found, skipping.")
				return []

		imgPath = imgObject.path
		rgbImg = imgObject.getRGB()
		if rgbImg is None:
			raise Exception("Unable to load image: {}".format(imgPath))

		bbs = self.align.getAllFaceBoundingBoxes(rgbImg)
		if bbs is None:
	 		raise Exception("Unable to find a face: {}".format(imgPath)) 
		
		
		
		for bb in bbs:
			start = time.time()
			alignedFace = self.align.align(self.imgDim, rgbImg, bb, 
				landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

			if alignedFace is None:
				raise Exception("Unable to align image: {}".format(imgPath))
			if self.verbose:
				print("  + Face alignment took {} seconds.".format(time.time() - start))

			# Stored aligned face into directory if we are training this mofo
			if isTrain:
				cv.imwrite(imgName,alignedFace)

			# Pass these reps through NN to get vec representation
			start = time.time()
			rep = self.net.forward(alignedFace)
			if self.verbose:
				print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
				print("Representation:")
				# print(rep)
				print("-----\n")
			reps.append((rep,bb)) 

		# Return the bb's and their vec representations. Rep[0]=bounding box Rep[1]=vector 
		return reps
	
	def rect_to_css(self,rect):
		"""
		Convert a dlib 'rect' object to a tuple(top,right,bottom,left)
		"""
		return rect.top(), rect.right(), rect.bottom(), rect.left()

	def markFace(self,frame,faces):
		"""
		Adds a bounding box with a label to an image.
		
		Params:
		frame - cv numpy array?
		faces - list of (string, float, rectangle)
		"""
		# Draw rectangle around faces
		for face in faces:
			name = face[0]
			confidence = face[1]
			(top,right,bottom,left) = self.rect_to_css(face[2]) # or is it (top,right,bottom,left?
			text = "{0} - confidence of {1:.2f}%".format(name,confidence*100)

			# Draw a box around the face
			cv.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)

			# Draw a label with a name below the face
			cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
			font = cv.FONT_HERSHEY_DUPLEX
			cv.putText(frame,text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		return frame


	# TODO Later, implement the following helper methods for modularity.
	# def preprocessImage(self,imgObject):
	# 	alignedFace = self.align(self.imgDim, imgObject, bb, 
	# 		landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

	
	# def extractFaces(self):
	# 	# Get bounding 
		
	# def alignFace(self,imgObject):
	# 	self.align(self.imgDim,imgObject)




