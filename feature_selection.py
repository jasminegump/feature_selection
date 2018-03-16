import re
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import sys
import random
import copy

# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

#col_names = ['c','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10', 'f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50']
col_names = ['c','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10']
#col_names = ['c','f1','f2','f3']

def read_file(text_file):
	data = []
	with open(text_file, 'rt') as f:
		for line in f:
			temp = re.sub(r'\n', '', line) # Remove all new lines
			temp = temp.split() # Split by tabs
			temp = [ float(num) for num in temp ]
			data.append(temp)
	return data

def read_input(text_file):
	input_list = read_file(text_file)
	df = list_to_pandas(input_list)
	return df

def list_to_pandas(input_list):
	df = pd.DataFrame(input_list, columns=col_names)
	#df = pd.DataFrame(input_list, columns=['c','f1','f2','f3'])
	return df

# need to normalize between -10 and 10
def calc_accuracy(num_pass, num_fail):
	return num_pass/(num_pass + num_fail)

def euclidean_distance(x1, x2):
	total = 0
	for i in range(len(x1)):
		total += (x1[i]-x2[i])**2
	return math.sqrt(total)

def get_neighbors(train_set, test):
	distance = []

	# because first column is class
	for i in range(len(train_set)):
		x1 = []
		x2 = []
		for j in range(1,len(test)):
			x1.append(test[j])
		for k in range(1,len(test)):
			x2.append(train_set[i][k])
		#print("x1",x1)
		#print("x2",x2)
		dist = euclidean_distance(x1,x2)
		distance.append(dist)
	idx = np.argmin(distance)
	return train_set[idx]

def z_normalize_df(df):
	#https://stackoverflow.com/questions/24761998/pandas-compute-z-score-for-all-columns
	for i in df:
		if i is not 'c':
			#print(df[i], df[i].mean(), df[i].std())
			temp = (df[i] - df[i].mean())/df[i].std()
			df[i] = temp
			#print(df[i].sum())
	#print(df)
	return df

def min_max_normalize_df(df):
	for i in df:
		if i is not 'c':
			#print(df[i], df[i].mean(), df[i].std())
			temp = (df[i] - df[i].min())/(df[i].max()-df[i].min())
			df[i] = temp
			#print(df[i].sum())
	#print(df)
	return df

def scaling_normalize_df(df):
	for i in df:
		if i is not 'c':
			#print(df[i], df[i].mean(), df[i].std())
			temp = df[i]/100
			df[i] = temp
			#print(df[i].sum())
	#print(df)
	return df

def training(df, current_set,feature_to_add):
	train_set = []
	num_pass = 0
	num_fail = 0

	if feature_to_add == 0:
		list_of_features = [0]+current_set
	else:
		list_of_features = [0]+current_set+[feature_to_add]
	col_list = [col_names[i] for i in list_of_features]

	print("training:",col_list)

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
		#print(neighbors, test)
		if test[0] == neighbors[0]:
			num_pass +=1
		else:
			num_fail +=1
	accuracy = calc_accuracy(num_pass, num_fail)
	print(accuracy)
	return accuracy

# Current divorcing of cross validation
def leave_one_out_cross_validation(data, current_set, feature_to_add):
	#print("data:",data)
	#print("current_set",current_set)
	#print("feature_to_add",feature_to_add)
	accuracy = training(data, current_set,feature_to_add)
	#accuracy = random.uniform(0,1) 
	return accuracy

def forward_selection(data, num_features):

	current_set_of_features = []
	best_feature = []
	best_total_accuracy = 0 
	# loop through the features
	for i in range(1,num_features+1):
		print ("On level ",i, " of the search tree")
		best_so_far_accuracy = 0
		feature_to_add_at_this_level = []
		for j in range(1,num_features+1):
			if j not in current_set_of_features:
				print ("Considering adding the ",j, " feature")
				accuracy = leave_one_out_cross_validation(data, current_set_of_features,j) # RANDOM input for now cause it don't matter
				print("accuracy", accuracy, "best_so_far_accuracy", best_so_far_accuracy)
				if accuracy > best_so_far_accuracy:
					print("accuracy > best_so_far_accuracy")
					best_so_far_accuracy = accuracy
					if feature_to_add_at_this_level:
						feature_to_add_at_this_level = []
					feature_to_add_at_this_level.append(j)
		current_set_of_features += feature_to_add_at_this_level
		if (best_so_far_accuracy > best_total_accuracy) or (len(current_set_of_features) <= 3):
			#print("!!!!!!")
			best_feature = copy.deepcopy(current_set_of_features)
			best_total_accuracy = best_so_far_accuracy
		print("On level", i," I added feature ", feature_to_add_at_this_level,"to current set")
		print("current_set_of_features:", current_set_of_features)
		print("%%%%%%%%%%%%%%%%%%%%%%%%%")
	print("FINISHED", best_feature, best_total_accuracy)


def backwards_selection(data, num_features):
	total_results = []
	total_runs = 3
	run_diff = 0.01
	for l in range(0,total_runs):
		current_set_of_features = []
		best_feature = []
		best_total_accuracy = 0 
		removed_features = []
		for k in range(1,num_features+1):
			#print(k)
			current_set_of_features.append(k)


		# loop through the features
		for i in range(1,num_features+1):
			print ("On level ",i, " of the search tree")
			print("current_set_of_features:", current_set_of_features, len(current_set_of_features))
			best_so_far_accuracy = 0
			#current_set_of_features = []
			
			#best_so_far_accuracy = 0
			for j in range(1,num_features+1):
				features_to_test = copy.deepcopy(current_set_of_features)
				if j in current_set_of_features:
					print ("Considering removing the ",j, " feature")
					features_to_test.remove(j)
					accuracy = leave_one_out_cross_validation(data, features_to_test,0) # RANDOM input for now cause it don't matter
					print("accuracy", accuracy, "best_so_far_accuracy", best_so_far_accuracy)
					if (accuracy >= best_so_far_accuracy) or (accuracy >= best_so_far_accuracy-(run_diff*l)):
						print("accuracy < best_so_far_accuracy")
						best_so_far_accuracy = accuracy
						feature_to_remove = j
						best_total_accuracy = best_so_far_accuracy
					#feature_to_add_at_this_level.append(j)

			print("before", current_set_of_features)
			#if feature_to_remove in current_set_of_features:
			#print("feature_to_add_at_this_level",feature_to_add_at_this_level)
			current_set_of_features.remove(feature_to_remove)
			#current_set_of_features = feature_to_add_at_this_level
			print("removed", feature_to_remove)
			print("after", current_set_of_features)

			if (len(current_set_of_features) <= 3):
				#print("!!!!!!")
				best_feature = copy.deepcopy(current_set_of_features)
				break
				#best_total_accuracy = best_so_far_accuracy
			#print("On level", i," I added feature ", feature_to_add_at_this_level,"to current set")
			#print("current_set_of_features:", current_set_of_features)
			print("accuracy",best_total_accuracy)
			print("%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("FINISHED", best_feature, best_total_accuracy)
		total_results.append(best_feature)
	#print()	
	print("FINISHED", total_results)

def main():
	#df = read_input('super_small.txt')
	
	df = read_input('CS205_SMALLtestdata__35.txt')
	#df = read_input('CS205_BIGtestdata__2.txt')

	#df = scaling_normalize_df(df)
	#df = min_max_normalize_df(df)
	df = z_normalize_df(df)

	#training(df,[5,6,8],10)

	#forward_selection(df, len(col_names)-1)
	backwards_selection(df, len(col_names)-1)

main()
'''
small 35 = 6,3,2
large 2 = 1,3,24
'''

