import re
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import random

# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

col_names = ['c','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10']
#col_names = ['f1','f2','f3']

def read_file(text_file):
	data = []
	with open(text_file, 'rt') as f:
		for line in f:
			temp = re.sub(r'\n', '', line) # Remove all new lines
			temp = temp.split() # Split by tabs
			temp = [ float(num) for num in temp ]
			data.append(temp)
	return data

def list_to_pandas(input_list):
	#df = pd.DataFrame(input_list, columns=['c','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])
	df = pd.DataFrame(input_list, columns=['c','f1','f2','f3'])
	return df

def euclidean_distance(x1, x2):
	total = 0
	for i in range(len(x1)):
		total += (x1[i]-x2[i])**2
	return math.sqrt(total)
	#return 1
	#return math.sqrt((x1-x2)**2)

def new_euclidean_distance(arg):
	return math.sqrt((x1-x2)**2)

def get_neighbors(train_set, test):
	distance = []
	print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	print(test[0],test[1],test[2])
	print(train_set[0][1],train_set[0][2])

	# because first column is class
	for i in range(len(train_set)):
		x1 = []
		x2 = []
		for j in range(1,len(test)):
			x1.append(test[j])
		for k in range(1,len(test)):
			x2.append(train_set[i][k])
		print("x1",x1)
		print("x2",x2)
		dist = euclidean_distance(x1,x2)
		distance.append(dist)
	idx = np.argmin(distance)
	return train_set[idx]

def calc_accuracy(num_pass, num_fail):
	return num_pass/(num_pass + num_fail)

def read_input(text_file):
	input_list = read_file(text_file)
	df = list_to_pandas(input_list)
	return df

def training(df, current_set,feature_to_add):
	train_set = []
	num_pass = 0
	num_fail = 0

	list_of_features = [0]+current_set+feature_to_add
	col_list = [col_names[i] for i in list_of_features]
	#print(col_list)

	# first create training data set with current set and feature to add
	#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
	for index,row in df.iterrows():
		row_data = []
		for i in col_list:
			#print (row[i])
			row_data.append(row[i]) 
		train_set.append(row_data)
	#print(train_set)

	for i in range(len(train_set)):
		temp = copy.deepcopy(train_set)
		test = train_set[i]
		#print(train_set[i])
		temp[i] = [1.0]
		for j in range(len(col_list)-1):
			temp[i].append(sys.maxsize)

		neighbors = get_neighbors(temp,test)

		if test[0] == neighbors[0]:
			num_pass +=1
		else:
			num_fail +=1
	accuracy = calc_accuracy(num_pass, num_fail)
	print(accuracy)
	'''
	train_set = []

	
	
	for col in range(len(col_list)):
		#print(col_list[col])	
		num_pass = 0
		num_fail = 0

		for i in range(len(df)):
			temp = [df['c'][i],df[col_list[col]][i]]
			train_set.append(temp)

		for i in range(len(train_set)):
			temp = copy.deepcopy(train_set)
			test = train_set[i]
			temp[i] = [1.0,sys.maxsize]

			neighbors = get_neighbors(temp,test)

			if test[0] == neighbors[0]:
				num_pass +=1
			else:
				num_fail +=1
		accuracy = calc_accuracy(num_pass, num_fail)
		print(accuracy)
		train_set = []
		'''
	

# Current divorcing of cross validation
def leave_one_out_cross_validation(data, current_set, feature_to_add):
	print("data:",data)
	print("current_set",current_set)
	print("feature_to_add",feature_to_add)
	accuracy = random.uniform(0,1) 
	return accuracy

def forward_selection(data, num_features):

	current_set_of_features = []

	# loop through the features
	for i in range(num_features):
		print ("On level ",i, " of the search tree")
		best_so_far_accuracy = 0
		feature_to_add_at_this_level = []
		for j in range(num_features):
			if j not in current_set_of_features:
				print ("Considering adding the ",j, " feature")
				accuracy = leave_one_out_cross_validation(data, current_set_of_features,j) # RANDOM input for now cause it don't matter
				print("accuracy", accuracy, "best_so_far_accuracy", best_so_far_accuracy)
				if accuracy > best_so_far_accuracy:
					print("accuracy > best_so_far_accuracy")
					best_so_far_accuracy = accuracy
					feature_to_add_at_this_level.append(j)
		current_set_of_features += feature_to_add_at_this_level
		print("On level", i," I added feature ", feature_to_add_at_this_level,"to current set")
		print("current_set_of_features:", current_set_of_features)
		print("%%%%%%%%%%%%%%%%%%%%%%%%%")

def main():
	df = read_input('super_small.txt')
	#df = read_input('CS205_SMALLtestdata__65.txt')
	
	training(df,[1],[2])
	#forward_selection(df, len(col_list))


main()






'''
f1
0.85
f2
0.79
f3
0.77
f4
0.76
f5
0.78
f6
0.72
f7
0.69
f8
0.84
f9
0.74
f10
0.74
'''

	





