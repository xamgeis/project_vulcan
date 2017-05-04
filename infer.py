import argparse
import cv2 as cv
import os
import pickle
import sys
import time
from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.lda import LDA
from sklearn.mixture import GMM
from sklearn.pipeline import Pipeline


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')

imgDim = 96 # TODO make this a global parameter
# Check that we have generated embeddings: labels.csv and reps.csv

def getRep(imgPath, multiple=False, verbose=False):
	"""
	Detects faces in the given image path. If faces are found, then they are 
	aligned and passed through the network to get a vector representation. 
	Returns a list of tuples of type (numpy.darray, int) 
	"""
	start = time.time()
	bgrImg = cv.imread(imgPath)
	if bgrImg is None:
		raise Exception("Unable to load image: {}".format(imgPath))

	rgbImg = cv.cvtColor(bgrImg, cv.COLOR_BGR2RGB)

	if verbose:
		print(" + Original size: {}".format(rgbImg.shape))
	if verbose:
		print("Loading the image took {} seconds.".format(time.time() - start))

	start = time.time()

	if multiple:
		bbs = align.getAllFaceBoundingBoxes(rgbImg)
	else:
		bb1 = align.getLargestFaceBoundingBox(rgbImg)
		bbs = [bb1]
	if len(bbs) == 0 or (not multiple and bb1 is None):
		raise Exception("Unable to find a face: {}".format(imgPath))
	if verbose:
		print("Face detection took {} seconds.".format(time.time() - start))

	reps = []
	for bb in bbs:
		start = time.time()
		alignedFace = align.align(
			imgDim,
			rgbImg,
			bb,
			landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
		if alignedFace is None:
			raise Exception("Unable to align image: {}".format(imgPath))
		if verbose:
			print("Alignment took {} seconds.".format(time.time() - start))
			print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

		start = time.time()
		rep = net.forward(alignedFace)
		if verbose:
			print("Neural network forward pass took {} seconds.".format(
				time.time() - start))
		reps.append((bb.center().x, rep))
	sreps = sorted(reps, key=lambda x: x[0])
	return sreps

def train(workDir,ldaDim=1):
	"""
	Trains a linear classifier on the recognized faces.
	"""
	print("Loading embeddings.")
	fname = "{}/labels.csv".format(workDir)
	labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
	labels = map(itemgetter(1),
				 map(os.path.split,
					 map(os.path.dirname, labels))) # Get the directory.
	fname = "{}/reps.csv".format(workDir)
	embeddings = pd.read_csv(fname, header=None).as_matrix()
	le = LabelEncoder().fit(labels)
	labelsNum = le.transform(labels)
	nClasses = len(le.classes_)
	print("Training for {} classes.".format(nClasses))

	# Linear SVM classifier
	clf = SVC(C=1, kernel='linear', probability=True)

	if ldaDim > 0:
		clf_final = clf
		clf = Pipeline([('lda', LDA(n_components=ldaDim)),
						('clf', clf_final)])

	clf.fit(embeddings, labelsNum)

	# save classifer in package
	fName = "{}/classifier.pkl".format(workDir)
	print("Saving classifier to '{}'".format(fName))
	with open(fName, 'w') as f:
		pickle.dump((le, clf), f)

def infer(classifierModel, imgs,multiple=False,verbose=False):
	"""
	Calls a function to extract and process faces in an image, then predicts with 
	some confidence what face exists in the image.
	@Return: none
	"""
	with open(classifierModel, 'rb') as f:
		if sys.version_info[0] < 3:
				(le, clf) = pickle.load(f)
		else:
				(le, clf) = pickle.load(f, encoding='latin1')

	for img in imgs:
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

if __name__ == '__main__':
	start = time.time()
	dlibFacePredictor = os.path.join(dlibModelDir,"shape_predictor_68_face_landmarks.dat")

	align = openface.AlignDlib("models/dlib/shape_predictor_68_face_landmarks.dat")
	net = openface.TorchNeuralNet(networkModel, imgDim, cuda=False) # model used for formatting

	print("Loading the dlib and OpenFace models took {} seconds.".format(
		time.time() - start)) 
	start = time.time()

	parser = argparse.ArgumentParser()
	parser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")
	args = parser.parse_args()
	classifierModel = "./generated-embeddings/classifier.pkl"
	workDir = "./generated-embeddings/"
	# train(workDir)
	infer(classifierModel,args.imgs, multiple=False, verbose=True)