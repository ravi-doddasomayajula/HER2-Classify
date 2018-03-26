import os, sys
import random
import numpy as np
import cv2
import scipy
import csv
import pandas as pd
import scipy.misc as smisc
from keras.metrics import categorical_accuracy

dirnmFD = '/home/diam/Desktop/Testing_fixed/Aperio_FDA'
lsFD = os.listdir(dirnmFD)
lsFD.sort()
dirnmND = '/home/diam/Desktop/Testing_fixed/Aperio_NIH'
lsND = os.listdir(dirnmND)
lsND.sort()
dirnmH2 = '/home/diam/Desktop/Testing_fixed/Hamamatsu2'
lsH2 = os.listdir(dirnmH2)
lsH2.sort()

## Read in Images ####
#FDA
X_FDA = []
idx_FDA = []
for index, image_filename in list(enumerate(lsFD)):
	img_file = cv2.imread(dirnmFD + '/' + image_filename)
	if img_file is not None:
		img_file = smisc.imresize(arr = img_file, size = (600,760,3))
		#img_file = smisc.imresize(arr = img_file, size = (120,160,3))		
		img_arr = np.asarray(img_file)
		X_FDA.append(img_arr)
		idx_FDA.append(index)

X_FDA = np.asarray(X_FDA)
idx_FDA = np.asarray(idx_FDA)
#random.seed(rs)
random_id = random.sample(idx_FDA, len(idx_FDA)/2)
random_FDA = []
for i in random_id:
	random_FDA.append(X_FDA[i])

random_FDA = np.asarray(random_FDA)
#NIH
X_NIH = []
idx_NIH = []
#nm_NIH = []
for index, image_filename in list(enumerate(lsND)):
	img_file = cv2.imread(dirnmND + '/' + image_filename)
	if img_file is not None:
		img_file = smisc.imresize(arr = img_file, size = (600,760,3))		
		#img_file = smisc.imresize(arr = img_file, size = (120,160,3))
		img_arr = np.asarray(img_file)
		#nm_NIH.append(image_filename)
		X_NIH.append(img_arr)
		idx_NIH.append(index)

X_NIH = np.asarray(X_NIH)
idx_NIH = np.asarray(idx_NIH)
random_NIH = []
for i in random_id:
	random_NIH.append(X_NIH[i])

random_NIH = np.asarray(random_NIH)

X_H2 = []
idx_H2 = []
for index, image_filename in list(enumerate(lsH2)):
	img_file = cv2.imread(dirnmH2 + '/' + image_filename)
	if img_file is not None:
		img_file = smisc.imresize(arr = img_file, size = (600,760,3))		
		#img_file = smisc.imresize(arr = img_file, size = (120,160,3))
		img_arr = np.asarray(img_file)
		X_H2.append(img_arr)
		idx_H2.append(index)

X_H2 = np.asarray(X_H2)
idx_H2 = np.asarray(idx_H2)
random_H2 = []
for i in random_id:
	random_H2.append(X_H2[i])

random_H2 = np.asarray(random_H2)

## Combine full sets from 3 scanners to form full main set ####
full_set = np.concatenate((X_FDA,X_NIH,X_H2),axis = 0)
## Combine random sets from 3 scanners to form training set ####
train_set = np.concatenate((random_FDA,random_NIH,random_H2),axis = 0)
X_train = train_set

## Test set ####
#l = set(full_set) - set(train_set)
l = set(idx_H2)-set(random_id)
#print('test set length is 3*',len(l))
test_H2 = []
test_FDA = []
test_NIH = []

for i in l:
	tst_file1 = X_H2[i]
	test_H2.append(tst_file1)	
	tst_file2 = X_NIH[i]
	test_NIH.append(tst_file2)	
	tst_file3 = X_FDA[i]
	test_FDA.append(tst_file3)

test_H2 = np.asarray(test_H2)
test_NIH = np.asarray(test_NIH)
test_FDA = np.asarray(test_FDA)

test_set = np.concatenate((test_FDA,test_NIH,test_H2),axis = 0)
X_test = test_set

## Scores/Labels ####		

score_all = []
score_file = open("scores","r")

for line in score_file:
	#line = line.replace("'","")	
	score_all.append(line)


scores = []
i_score = []

for ind, score in list(enumerate(score_all)):
	scores.append(score.strip('\n'))
	i_score.append(ind)


#scoresf = scores
scoresf = []

for i in scores:
#	scoresf.append(float(i))
	scoresf.append(int(i))

random_score_FDA = []

for i in random_id:
	random_score_FDA.append(scoresf[i])
	

random_score_FDA = np.asarray(random_score_FDA)

random_score_NIH = []

for i in random_id:
	random_score_NIH.append(scoresf[i])
	

random_score_NIH = np.asarray(random_score_NIH)

random_score_H2 = []

for i in random_id:
	random_score_H2.append(scoresf[i])
	

random_score_H2 = np.asarray(random_score_H2)

truth_score = np.concatenate((random_score_FDA,random_score_NIH,random_score_H2),axis = 0)
y_train = truth_score
#y_dummy = np_utils.to_categorical(y_train)

## test scores ####

test_sc_H2 = []
test_sc_FDA = []
test_sc_NIH = []

for i in l:
	tst_1 = scoresf[i]
	test_sc_H2.append(tst_1)	
	tst_2 = scoresf[i]
	test_sc_NIH.append(tst_2)	
	tst_3 = scoresf[i]
	test_sc_FDA.append(tst_3)


test_sc_H2 = np.asarray(test_sc_H2)
test_sc_NIH = np.asarray(test_sc_NIH)
test_sc_FDA = np.asarray(test_sc_FDA)

test_sc_set = np.concatenate((test_sc_FDA,test_sc_NIH,test_sc_H2),axis = 0)
y_test = test_sc_set
#y_test_dummy = np_utils.to_categorical(y_test)

## ENCODING LABELS ####
encoder = LabelEncoder()
y_train1 = encoder.fit_transform(y_train)
y_test1 = encoder.fit_transform(y_test)
y_train_dummy = np_utils.to_categorical(y_train1)
y_test_dummy = np_utils.to_categorical(y_test1)

## Flip train and test sets ####
X_train_new = X_train
X_test_new = X_test
y_train_dummy_new = y_train_dummy
y_test_dummy_new = y_test_dummy






#trainSet = np.add(random_FDA, random_NIH, random_H2)

#tfile1 = open('test1.txt', 'w')
#for i in random_id:
#	print>>tfile1, lsFD[i]

#tfile2 = open('test2.txt', 'w')
#for i in random_id:
#	print>>tfile2, lsND[i]



#for caseFD in randFD
#	str = caseFD[:-6:]
#	lsND = 
#print(randn1[1][-6::])


