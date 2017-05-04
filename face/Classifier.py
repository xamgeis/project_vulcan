import cv2 as cv
import os
import pickle
import sys

from operator import itemgetter

import numpy as np

np.set_printoptions(precision=2)

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

class Classifier:
	fileDir = os.path.dirname(os.path.realpath(__file__))
	modelDir = os.path.join(fileDir, 'models')
	dlibModelDir = os.path.join(modelDir, 'dlib')
	openfaceModelDir = os.path.join(modelDir, 'openface')


	def __init__(self):
		print ""

	def train():
		return ""

	def infer(classifierModelDir, reps,multiple=False,verbose=False):
		"""
		Calls a function to extract and process faces in an image, then predicts with 
		some confidence what face exists in the image.
		@Return: none
		"""
		with open(calssifierModelDir, 'rb') as f:
			if sys.version_info[0] < 3:
					(le, clf) = pickle.load(f)
			else:
					(le, clf) = pickle.load(f, encoding='latin1')

		for img in reps:
			print("\n=== {} ===".format(img))
			reps = getRep(img, multiple)
			if len(reps) > 1:
				print("List of faces in image from left to right")
			for r in reps:
				rep = r[1].reshape(1, -1)
				bbx = r[0]
				start = time.time()
				predictions = clf.predict_proba(rep).ravel()
				maxI = np.argmax(predictions)
				person = le.inverse_transform(maxI)
				confidence = predictions[maxI]
				if verbose:
					print("Prediction took {} seconds.".format(time.time() - start))
				if multiple:
					print("Predict {} @ x={} with {:.2f} confidence.".format(person.decode('utf-8'), bbx,
																			 confidence))
				else:
					print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
				if isinstance(clf, GMM):
					dist = np.linalg.norm(rep - clf.means_[maxI])
					print(" + Distance from the mean: {}".format(dist))