import cv2 as cv
import os
import pickle
import sys
import pandas as pd

import numpy as np
np.set_printoptions(precision=2)

from operator import itemgetter
from Util import Util



import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

class Classifier:

	def __init__(self,embeddingsDir=""):
		self.util = Util()
		if embeddingsDir == "":
			self.embeddingsDir = self.util.embeddingsDir
		else:
			self.embeddingsDir = embeddingsDir
		print ""

	def train(self):
		# Load embeddings from labels.csv and reps.csv 
		print("Loading embeddings.")
		fname = "{}/labels.csv".format(self.embeddingsDir) 
		labels = pd.read_csv(fname, header=None).as_matrix()[:,1]
		labels = map(itemgetter(1), 
								map(os.path.split, 
									map(os.path.dirname, labels))) # Get the directory
		fname = "{}/reps.csv".format(self.embeddingsDir)
		embeddings = pd.read_csv(fname, header=None).as_matrix()

		# Clean the data by encoding 
		le = LabelEncoder().fit(labels)
		labelsNum = le.transform(labels)
		nClasses = len(le.classes_)
		print("Training for {} classes.").format(nClasses)
		
		# initialize our LinearSVM classifier
		clf = SVC(C=1, kernel='linear',probability=True)
			# use LDA pipeline?
		# fit the classifier with embeddings and labels
		clf.fit(embeddings,labelsNum)	
		# Save classifier to disk
		fname = "{}/classifier.pkl".format(self.embeddingsDir)
		print("Saving classifier to '{}'").format(fname)
		with open(fname, 'w') as f:
			pickle.dump((le,clf), f)


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