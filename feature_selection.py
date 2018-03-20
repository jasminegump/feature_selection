import re
import pandas as pd
import math
import numpy as np
import copy
import sys
import random
import copy
import time

#data mining text book
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

col_names = ['c','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10', 'f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50']

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
	df, features_total = list_to_pandas(input_list)
	return df, features_total

def list_to_pandas(input_list):
	# get variable number of columns
	#print(len(input_list[0]))
	feature_total = len(input_list[0])
	data_total = len(input_list)
	print("This dataset has ", feature_total - 1, "features (not including the class attribute) with ", data_total," instances.\n")
	df = pd.DataFrame(input_list, columns=col_names[:feature_total])
	return df, feature_total

# need to normalize between -10 and 10
def calc_accuracy(num_pass, num_fail):
	return num_pass/(num_pass + num_fail)

def euclidean_distance(x1, x2):
	total = 0
	#print(x1,x2)
	for i in range(len(x1)):
		#print(x1,x2,x1[i],x2[i])
		total += (x1[i]-x2[i])**2
	return math.sqrt(total)

def get_neighbors(train_set, test):
	distance = []
	#print(test)
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

def new_min_max_normalize_df(df):
	new_min = 0
	new_max = 1

	for i in df:
		if i is not 'c':
			#print(df[i], df[i].mean(), df[i].std())
			temp = (df[i] - df[i].min())/(df[i].max()-df[i].min())*(new_max-new_min) + new_min
			df[i] = temp
			#print(df[i].sum())
	#print(df)
	return df

def z_normalize_df(df):
	#https://stackoverflow.com/questions/24761998/pandas-compute-z-score-for-all-columns
	for i in df:
		if i is not 'c':
			#print(df[i], df[i].mean(), df[i].std())
			temp = (df[i] - df[i].mean())/df[i].std()
			df[i] = temp
			#print(df[i].sum())
	#print(df)
	print("Please wait while I normalize the data... Done!\n")
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

	#print("training:",col_list)

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
		#print(temp,test)
		neighbors = get_neighbors(temp,test)
		#print(neighbors, test)
		if test[0] == neighbors[0]:
			num_pass +=1
		else:
			num_fail +=1
	accuracy = calc_accuracy(num_pass, num_fail)
	#print(accuracy)
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
	found_all = 0
	# loop through the features
	for i in range(1,num_features):
		print ("On level ",i, " of the search tree")
		best_so_far_accuracy = 0
		feature_to_add_at_this_level = []
		for j in range(1,num_features):
			if j not in current_set_of_features:
				accuracy = leave_one_out_cross_validation(data, current_set_of_features,j)
				test_set = current_set_of_features + [j]
				print("Using feature(s)",test_set, "accuracy is", accuracy*100, "%")
				if (accuracy > best_so_far_accuracy):
					best_so_far_accuracy = accuracy
					if feature_to_add_at_this_level:
						feature_to_add_at_this_level = []
					feature_to_add_at_this_level.append(j)
		current_set_of_features += feature_to_add_at_this_level
		print("Features set", current_set_of_features, "was best, accuracy is", best_so_far_accuracy*100, "%")
		if best_so_far_accuracy <= best_total_accuracy:
			if found_all == 0:
				print("Warning, accuracy has decreased! Continuing search in case of local maxima")
			found_all = 1
		if (best_so_far_accuracy > best_total_accuracy) and not found_all:
			best_feature = copy.deepcopy(current_set_of_features)
			best_total_accuracy = best_so_far_accuracy
		print("\n")
	print("Finished search!!! The best feature subset is ", best_feature, "which has an accuracy of", best_total_accuracy*100, "%")
	return best_feature, best_total_accuracy

def backwards_selection(data, num_features):

	current_set_of_features = []
	best_feature = []
	best_total_accuracy = 0 
	removed_features = []
	found_all = 0

	for k in range(1,num_features):
		current_set_of_features.append(k)


	# loop through the features
	for i in range(1,num_features):
		print ("On level ",i, " of the search tree")
		best_so_far_accuracy = 0

		for j in range(1,num_features):
			features_to_test = copy.deepcopy(current_set_of_features)
			if j in current_set_of_features:
				#print ("Considering removing the ",j, " feature")
				features_to_test.remove(j)
				accuracy = leave_one_out_cross_validation(data, features_to_test,0) # RANDOM input for now cause it don't matter
				print("Using feature(s)",features_to_test, "accuracy is", accuracy*100, "%")
				if (accuracy >= best_so_far_accuracy):
					#print("accuracy < best_so_far_accuracy")
					best_so_far_accuracy = accuracy
					feature_to_remove = j
				#feature_to_add_at_this_level.append(j)
		if accuracy >= best_so_far_accuracy and (j > 1):
			print("Warning, accuracy has decreased! Continuing search in case of local maxima")

		current_set_of_features.remove(feature_to_remove)
		print("Features set", current_set_of_features, "was best, accuracy is", best_so_far_accuracy*100, "%")


		if best_so_far_accuracy >= best_total_accuracy:
			best_feature = copy.deepcopy(current_set_of_features)
			best_total_accuracy = best_so_far_accuracy

		print("\n")
	print("Finished search!!! The best feature subset is ", best_feature, "which has an accuracy of", best_total_accuracy*100, "%")


def jasmine_search_algorithm(data, num_features):

	features = []
	total_results = []
	total_runs = 3
	run_diff = 0.02
	orig_data = copy.deepcopy(data)
	untouched_original = copy.deepcopy(data)
	for k in range(0,total_runs):
		for l in range(0,total_runs):
			data = orig_data
			#delete random 5% of data
			# resample 5%
			resample = int(len(data)/100*40)
			#print(len(data))
			for m in range(0,resample):
				temp_rand = random.randint(0,len(data)-1)
				#print(temp_rand)
				#https://chrisalbon.com/python/data_wrangling/pandas_dropping_column_and_rows/
				data = data.drop(data.index[temp_rand])
			#print(len(data))

			best_feature, best_total_accuracy = forward_selection(data, num_features)
			total_results.append(best_feature)
		
		strong_feature = find_strong_feature(total_results)
		
		s_feature = col_names[strong_feature]
		s_feature = s_feature[1:]
		features.append(int(s_feature))
		print("")
		print("Strongest Feature is", s_feature, ", hence removing it...")
		print("")
		#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html
		temp = col_names[strong_feature]
		data = data.drop([temp], axis=1)
		orig_data = orig_data.drop([temp], axis=1)
		#print(data)
		col_names.remove(temp)
		num_features -= 1
	print("FEATURES:", features)

	temp_final = []
	final_a = []
	for n in range(0,total_runs):
		temp_final.append(features[n])
		best_total_accuracy = training(untouched_original,temp_final,0)
		final_a.append(best_total_accuracy)
		print("temp_final",temp_final, "accuracy",best_total_accuracy)
	max_accuracy = max(final_a)
	final_set = features[:final_a.index(max_accuracy)+1]
	print("Finished search!!! The best feature subset is ", final_set, "which has an accuracy of", max_accuracy*100, "%")
	return total_results



	#best_total_accuracy = training(untouched_original,features,0)
	#print("Finished search!!! The best feature subset is ", features, "which has an accuracy of", best_total_accuracy*100, "%")
	#return total_results

def find_strong_feature(feature_list):
	count = 0
	strong_features = dict()
	#features_output = []
	for i in range(len(feature_list)):
		for j in range(len(feature_list[i])):
			if feature_list[i][j] in strong_features:
				strong_features[feature_list[i][j]] = strong_features[feature_list[i][j]] + 1
			else:
				strong_features[feature_list[i][j]] = 1
	#print (strong_features)
	
	#https://stackoverflow.com/questions/26871866/print-highest-value-in-dict-with-key
	max_key = max(strong_features, key=strong_features.get)  # Just use 'min' instead of 'max' for minimum.
	features_output = max_key
	max_key = strong_features.pop(max_key)
	#max_key = max(strong_features, key=strong_features.get)  # Just use 'min' instead of 'max' for minimum.
	#features_output.append(max_key)

	return features_output


def find_default(df, features_total):
	current_set = []
	current_set = [i for i in range(1,features_total)]
	accuracy = training(df, current_set,0)
	return accuracy

def main():


	print("Welcome to Jasmine's Feature Selection Algorithm\n")

	# FOR DEBUGGING REASONS THIS IS COMMENTED OUT
	text_file = input("Type in the name of the file to test:")
	#text_file = 'CS205_SMALLtestdata__35.txt'
	#text_file = 'CS205_BIGtestdata__2.txt'

	df,features_total = read_input(text_file)

	print("Type the number of the algorithm you want to run")
	print("1. Forward Selection")
	print("2. Backward Elimination")
	print("3. Jasmine's Search Algorithm")
	algorithm = int(input())

	start = time.time()

	features = []

	accuracy = find_default(df, features_total)

	df = z_normalize_df(df)

	find_default(df, features_total)

	temp = training(df, [6,2,3], 0)
	print("ACCURACY",temp)

	accuracy_percentage = accuracy*100
	print("Running nearest neighbor with all ", features_total-1, "features, using 'leaving-one-out' evaluation, I get an accuracy of ", accuracy_percentage, "%")

	print("Beginning search.")

	if algorithm == 1:
		forward_selection(df, features_total)
	elif algorithm == 2:
		backwards_selection(df, features_total)
	elif algorithm == 3:
		jasmine_search_algorithm(df, features_total)
	else:
		print("Invalid")

	end = time.time()
	print("Time lapsed (seconds): ", end-start)

main()

