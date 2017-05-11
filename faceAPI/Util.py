import os 

class Util():
	fileDir = os.path.dirname(os.path.realpath(__file__))
	modelDir = os.path.join(fileDir, 'models')
	dlibModelDir = os.path.join(modelDir, 'dlib')
	openfaceModelDir = os.path.join(modelDir, 'openface')
	alignedImgsDir = os.path.join(fileDir,'aligned-images')
	networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
	embeddingsDir = os.path.join(fileDir, 'generated-embeddings')
	classifierModelDir = os.path.join(embeddingsDir,'classifier.pkl')
	
	# batch_representDir = os.path.join(fileDir,'classifier.pkl')

	def __init__(self):
		# do nothing
		a = 1+1
