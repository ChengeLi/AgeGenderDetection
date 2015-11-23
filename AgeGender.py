import os
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import sklearn 


if __name__ == '__main__':
	data         = pickle.load( open( "../data/Fam2a_file", "rb" ) )
	data_face    = np.array(data['face'])
	data_feature = np.array(data['feature'])
	data_age     = np.array(data['age'])
	data_sex     = np.array(data['sex'])

	numSample      = data_age.shape[0]
	index_data     = range(numSample)
	np.random.shuffle(index_data)
	index_training = index_data[:2*numSample/3]
	index_testing  = index_data[2*numSample/3:]
	
	feature_training = data_feature[index_training,:]
	sex_training     = data_sex[index_training]
	
	feature_testing  = data_feature[index_testing,:]
	sex_testing      = data_sex[index_testing]




	# SVM
	clf        = svm.SVC(gamma='auto')
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
	clf        = grid_search.GridSearchCV(svr, parameters)
	clf.fit(feature_training, sex_training)  
	sex_test_result = clf.predict(feature_testing)



