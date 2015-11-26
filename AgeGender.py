import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import glob as glob
from sklearn import svm


if __name__ == '__main__':
	
	files        = sorted(glob.glob('../data/'+'*_file*'))
	dataFromFile = {}
	data_face    = []
	data_feature = []
	data_age     = []
	data_sex     = []
	for ii in range(len(files)):
		dataFromFile[ii] = pickle.load( open( files[ii], "rb" ) )
		# data         = pickle.load( open( "../data/Fam2a_file", "rb" ) )
		data = dataFromFile[ii]
		data_face.append(np.array(data['face']))
		data_feature.append(data['feature'])
		data_age.append(data['age'])
		data_sex.append(data['sex'])


	data_face_stack    = np.array(data_face[0])
	data_feature_stack = np.array(data_feature[0])
	data_age_stack     = np.array(data_age[0])
	data_sex_stack     = np.array(data_sex[0])

	for kk in range(len(files)-1):
		data_face_stack    = np.concatenate((data_face_stack,np.array(data_face[kk+1])), axis=0)
		data_feature_stack = np.concatenate((data_feature_stack,np.array(data_feature[kk+1])), axis=0)
		data_age_stack     = np.concatenate((data_age_stack,np.array(data_age[kk+1])), axis=0)
		data_sex_stack     = np.concatenate((data_sex_stack,np.array(data_sex[kk+1])), axis=0)



	numSample      = data_face_stack.shape[0]
	index_data     = range(numSample)
	np.random.shuffle(index_data)
	index_training = index_data[:1*numSample/4]
	index_testing  = index_data[3*numSample/4:]
	
	feature_training = data_feature_stack[index_training,:]
	sex_training     = data_sex_stack[index_training]
	
	feature_testing  = data_feature_stack[index_testing,:]
	sex_testing      = data_sex_stack[index_testing]


	# SVM
	clf        = svm.SVC(kernel = 'linear')
	# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
	# clf        = grid_search.GridSearchCV(svr, parameters)
	clf.fit(feature_training, sex_training)  
	sex_test_result = clf.predict(feature_testing)
	sex_train_result = clf.predict(feature_training)



