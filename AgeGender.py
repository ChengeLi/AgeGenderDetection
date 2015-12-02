import os
import pdb
import numpy as np

import glob as glob
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

import cPickle as pickle
import matplotlib.pyplot as plt



def PCAembedding():
	# project original data into lower dimensions
	pass


def preprocess(X,y,test_size):
	print "preprocessing..."
	# preprocessing and Cross Validation
	X = StandardScaler().fit_transform(X)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size)
	return X_train,X_test,y_train,y_test





def GenderClassifier(data_feature_stack,data_gender_stack,test_size = 0.5,search = False):
	genderX_train,genderX_test,genderY_train,genderY_test = preprocess(data_feature_stack,data_gender_stack,test_size)
	
	print "fitting gender Clssfifer..."
	"""grid search for best C"""
	if search:
		clf = svm.SVC(kernel = 'linear' )
		parameters = {'kernel':['linear'], 'C':[0.0001,0.001,0.005]} #among 0.0001,0.001, 0.01, 0.1, 1,10,100, 0.001 is the best
		cv_clf = grid_search.GridSearchCV(clf, parameters)
		cv_clf.fit(genderX_train, genderY_train)
		# np.mean(cross_validation.cross_val_score(cv_clf.fit(genderX_train).best_estimator_, genderX_train))
		print "cv_clf.best_params_: ",cv_clf.best_params_
		clf = cv_clf
	else:	
		clf = svm.SVC(kernel = 'linear',C = 0.001)
		clf.fit(genderX_train, genderY_train)  
	
	print "predicting gender..."
	# gender_test_result  = clf.predict(genderX_test)
	# gender_train_result = clf.predict(genderX_train)
	
	gender_acc_test  = clf.score(genderX_test, genderY_test)
	gender_acc_train = clf.score(genderX_train, genderY_train)

	pdb.set_trace()

	#cross validation
	# scores = cross_validation.cross_val_score(clf, data_feature_stack, data_gender_stack, cv=5)

	return clf, gender_acc_test,gender_acc_train


def AgeClassifier(data_feature_stack,data_age_stack,test_size = 0.5):
	Age_range = np.unique(data_age_stack)
	# 923,  1529,   856,   1617,    13836,      6260,     1198

	AgeX_train,AgeX_test,AgeY_train,AgeY_test = preprocess(data_feature_stack,data_age_stack,test_size)
	print "fitting Age Clssfifer..."
	# parameters = (C=1.0, class_weight=None, dual=True, fit_intercept=True,\
	# intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',\
 #     random_state=0, tol=0.0001, verbose=0)

	clf = OneVsRestClassifier(LinearSVC(C = 0.001)).fit(AgeX_train, AgeY_train)


	print "predicting Age..."
	Age_test_result  = clf.predict(AgeX_test)
	Age_train_result = clf.predict(AgeX_train)	

	# Age_acc_test  = clf.score(AgeX_test, AgeY_test)
	# Age_acc_train = clf.score(AgeX_train, AgeY_train)
	Age_acc_test  = np.sum(Age_test_result == AgeY_test)
	Age_acc_train = np.sum(Age_train_result == AgeY_train)

	temp   = Age_test_result-AgeY_test
	error  = np.sqrt(temp**2)
	rmse   = np.mean(error)
	error2 = np.sqrt(temp[temp!=0]**2)
	rmse2  = np.mean(error2)


	pdb.set_trace()
	return clf, Age_acc_test,Age_acc_train


if __name__ == '__main__':
	
	files        = sorted(glob.glob('../data/'+'*_file*'))
	dataFromFile = {}
	data_face    = []
	data_feature = []
	data_age     = []
	data_gender  = []
	for ii in range(len(files)):
		dataFromFile[ii] = pickle.load( open( files[ii], "rb" ) )
		data = dataFromFile[ii]
		data_face.append(np.array(data['face']))
		data_feature.append(data['feature'])
		data_age.append(data['age'])
		data_gender.append(data['sex'])


	data_face_stack    = np.array(data_face[0])
	data_feature_stack = np.array(data_feature[0])
	data_age_stack     = np.array(data_age[0])
	data_gender_stack  = np.array(data_gender[0])

	for kk in range(len(files)-1):
		data_face_stack    = np.concatenate((data_face_stack,np.array(data_face[kk+1])), axis=0)
		data_feature_stack = np.concatenate((data_feature_stack,np.array(data_feature[kk+1])), axis=0)
		data_age_stack     = np.concatenate((data_age_stack,np.array(data_age[kk+1])), axis=0)
		data_gender_stack  = np.concatenate((data_gender_stack,np.array(data_gender[kk+1])), axis=0)


	
	numSample = data_face_stack.shape[0]
	
	"""Gender Clssfifer"""
	gender_clf, gender_acc_test,gender_acc_train = GenderClassifier(data_feature_stack,data_gender_stack,test_size =0.6)
	
	"""Age Clssfifer"""
	Age_clf, Age_acc_test,Age_acc_train = AgeClassifier(data_feature_stack,data_age_stack,test_size = 0.2)





"""
# random shuffle to get training and testing sets
	index_data     = range(numSample)
	np.random.shuffle(index_data)
	index_training = index_data[:1*numSample/4]
	index_testing  = index_data[3*numSample/4:]
	
	feature_training = data_feature_stack[index_training,:]
	sex_training     = data_sex_stack[index_training]
	
	feature_testing  = data_feature_stack[index_testing,:]
	sex_testing      = data_sex_stack[index_testing]
"""




















