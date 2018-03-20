# Use of helper libraries
import re
import pandas as pd
import math
import numpy as np
import copy
import sys
import random
import copy
import time

# Named columns to make it easier when debugging code to pinpoint
col_names = ['c','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10', 'f11','f12','f13','f14','f15','f16',\
'f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34', \
'f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50']

# Wrapper function to read in file and create pandas dataframe
def read_input(text_file):
	input_list = read_file(text_file)
	df, features_total = list_to_pandas(input_list)
	return df, features_total

# Reads in text file and converts text file into a nested list of float values
def read_file(text_file):
	data = []
	with open(text_file, 'rt') as f:
		for line in f:
			temp = re.sub(r'\n', '', line) # Remove all new lines
			temp = temp.split() # Split by spaces
			temp = [ float(num) for num in temp ]
			data.append(temp)
	return data

# Converts nested list into Pandas Data Frame
# Consulted: https://pandas.pydata.org/
def list_to_pandas(input_list):
	feature_total = len(input_list[0])
	data_total = len(input_list)
	print("This dataset has ", feature_total - 1, "features (not including the class attribute) with ", data_total," instances.\n")
	df = pd.DataFrame(input_list, columns=col_names[:feature_total])
	return df, feature_total

# Calculates accuracy
def calc_accuracy(num_pass, num_fail):
	return num_pass/(num_pass + num_fail)

# Calculates euclidean distance for two lists, x1 and x2
# Consulted:  https://docs.python.org/2/library/math.html and Data Mining Concepts and Techniques by Jiawei Han
def euclidean_distance(x1, x2):
	total = 0
	for i in range(len(x1)):
		total += (x1[i]-x2[i])**2
	return math.sqrt(total)

# Calculates z-score for each item per feature column
# Consulted: https://stackoverflow.com/questions/24761998/pandas-compute-z-score-for-all-columns
def z_normalize_df(df):
	for i in df:
		if i is not 'c':
			temp = (df[i] - df[i].mean())/df[i].std()
			df[i] = temp
	print("Please wait while I normalize the data... Done!\n")
	return df

# Takes in a list of of datapoints for the training and a test datapoint
# Finds and returns the nearest neighbor using Euclidean distance, to test datapoint, and training data set
def get_neighbors(train_set, test):
	distance = []

	# It loops through calculating the euclidean distance of every item in the training/test set
	# It also packs each datapoint into a list of features to calculate the euclidean distance between two multi-featured datapoints
	# It also picks the minimum of distance as the nearest neighbor
	for i in range(len(train_set)):
		x1 = []
		x2 = []
		for j in range(1,len(test)):
			x1.append(test[j])
		for k in range(1,len(test)):
			x2.append(train_set[i][k])
		dist = euclidean_distance(x1,x2)
		distance.append(dist)
	# Consulted: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.argmin.html
	idx = np.argmin(distance)
	return train_set[idx]

# This is the function that performs the actual leave-one-out cross validation
# Therefore, it is a wrapper function for the nearest neighbors and calculating the accuracy
def training(df, current_set,feature_to_add):
	train_set = []
	num_pass = 0
	num_fail = 0

	# Prepares the list of features to use in order to calculate euclidean distance
	if feature_to_add == 0:
		list_of_features = [0]+current_set
	else:
		list_of_features = [0]+current_set+[feature_to_add]
	col_list = [col_names[i] for i in list_of_features]

	# First create training data set with current set and feature to add
	#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
	for index,row in df.iterrows():
		row_data = []
		for i in col_list:
			row_data.append(row[i]) 
		train_set.append(row_data)

	# The following is where the actual leave-one-out cross validation occurs
	# It sets the training datapoint as all but the one test point
	# It then feeds in the training dataset and one test point into the nearest neighbors function
	for i in range(len(train_set)):
		temp = copy.deepcopy(train_set)
		test = train_set[i]
		temp[i] = [1.0] # This is hardcoded to a class of 1 but doesn't matter because it will never pick the maxsize
		for j in range(len(col_list)-1):
			temp[i].append(sys.maxsize) # This is hardcoded to fill in what's become the test set. But it doesn't matter because it will never pick this.
		neighbors = get_neighbors(temp,test)
		
		# Test to see if the nearest neighbors correctly classified test point
		if test[0] == neighbors[0]:
			num_pass +=1
		else:
			num_fail +=1
	# Calculate total accuracy with current dataset
	accuracy = calc_accuracy(num_pass, num_fail)
	return accuracy

# This function merely calls the training function and returns the accuracy of the training function
def leave_one_out_cross_validation(data, current_set, feature_to_add):
	accuracy = training(data, current_set,feature_to_add)
	return accuracy

# Searches for the strongest feature in the nested list
# Strongest feature is the feature that appears the most frequently in the nested list
# It returns the strongest feature
def find_strong_feature(feature_list):
	count = 0
	strong_features = dict()

	# This nested loop goes through creating a hash table of keys (potential strong features) and values (its occurence)
	for i in range(len(feature_list)):
		for j in range(len(feature_list[i])):
			if feature_list[i][j] in strong_features:
				strong_features[feature_list[i][j]] = strong_features[feature_list[i][j]] + 1
			else:
				strong_features[feature_list[i][j]] = 1

	# Finds the maximum key value in hash table
	# Consulted: https://stackoverflow.com/questions/26871866/print-highest-value-in-dict-with-key
	max_key = max(strong_features, key=strong_features.get)
	features_output = max_key
	max_key = strong_features.pop(max_key)

	return features_output

# Finds the default accuracy rate prior to any training
def find_default(df, features_total):
	current_set = []
	current_set = [i for i in range(1,features_total)]
	accuracy = training(df, current_set,0)
	return accuracy

# Consulted: http://www.cs.ucr.edu/~eamonn/205/Project_two_evaluation.pptx
# The following is the forward selection searching algorithm
def forward_selection(data, num_features):
	current_set_of_features = []
	best_feature = []
	best_total_accuracy = 0 
	found_all = 0

	# Loop through each level of the search tree
	for i in range(1,num_features):
		print ("On level ",i, " of the search tree")
		best_so_far_accuracy = 0
		feature_to_add_at_this_level = []
		# Loop through each feature at level
		for j in range(1,num_features):
			# Make sure we only consider adding the feature if it hasn't already been added before
			if j not in current_set_of_features:
				accuracy = leave_one_out_cross_validation(data, current_set_of_features,j)
				test_set = current_set_of_features + [j]
				print("Using feature(s)",test_set, "accuracy is", accuracy*100, "%")
				# Only consider if current accuracy is better than best so far
				if (accuracy > best_so_far_accuracy):
					best_so_far_accuracy = accuracy
					if feature_to_add_at_this_level:
						feature_to_add_at_this_level = []
					feature_to_add_at_this_level.append(j)
		current_set_of_features += feature_to_add_at_this_level
		print("Features set", current_set_of_features, "was best, accuracy is", best_so_far_accuracy*100, "%")
		# The following checks to see if accuracy has decreased
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

# The following is the backwards elimination searching algorithm
def backwards_elimination(data, num_features):

	current_set_of_features = []
	best_feature = []
	best_total_accuracy = 0 
	removed_features = []
	found_all = 0

	# This creates the initial state which is all features
	for k in range(1,num_features):
		current_set_of_features.append(k)

	# Loop through each level of the search tree
	for i in range(1,num_features):
		print ("On level ",i, " of the search tree")
		best_so_far_accuracy = 0
		# Loop through each feature at level
		for j in range(1,num_features):
			features_to_test = copy.deepcopy(current_set_of_features)
			# Make sure we only consider removing the feature if it hasn't already been removed before
			# Loop through, removing one feature at a time, calculating the accuracy
			# Remove the feature that provides you with the highest accuracy with its absence
			if j in current_set_of_features:
				features_to_test.remove(j)
				accuracy = leave_one_out_cross_validation(data, features_to_test,0) # RANDOM input for now cause it don't matter
				print("Using feature(s)",features_to_test, "accuracy is", accuracy*100, "%")
				if (accuracy >= best_so_far_accuracy):
					best_so_far_accuracy = accuracy
					feature_to_remove = j
		# The following checks to see if accuracy has decreased			
		if accuracy >= best_so_far_accuracy and (j > 1):
			print("Warning, accuracy has decreased! Continuing search in case of local maxima")
		# Remove the feature that provides you with the highest accuracy with its absence permanently from search space
		current_set_of_features.remove(feature_to_remove)
		print("Features set", current_set_of_features, "was best, accuracy is", best_so_far_accuracy*100, "%")
		if best_so_far_accuracy >= best_total_accuracy:
			best_feature = copy.deepcopy(current_set_of_features)
			best_total_accuracy = best_so_far_accuracy
		print("\n")

	print("Finished search!!! The best feature subset is ", best_feature, "which has an accuracy of", best_total_accuracy*100, "%")

# The following my own search algorithm
def jasmine_search_algorithm(data, num_features):

	features = []
	total_results = []
	total_runs = 3
	orig_data = copy.deepcopy(data)
	untouched_original = copy.deepcopy(data)

	# Go searching for the top strongest features
	for k in range(0,total_runs):

		# Create three copies of the dataset and resample
		for l in range(0,total_runs):
			data = orig_data
			# Resample - delete random 20% to 40% of data
			resample = int(len(data)/100*20)
			for m in range(0,resample):
				temp_rand = random.randint(0,len(data)-1)
				#Consulted : https://chrisalbon.com/python/data_wrangling/pandas_dropping_column_and_rows/
				data = data.drop(data.index[temp_rand])

			# Run forward selection on resampled data
			best_feature, best_total_accuracy = forward_selection(data, num_features)
			total_results.append(best_feature)
		
		# Find the strongest feature from the resampled data
		strong_feature = find_strong_feature(total_results)
		s_feature = col_names[strong_feature]
		s_feature = s_feature[1:]
		features.append(int(s_feature))
		print("")
		print("Strongest Feature is", s_feature, ", hence removing it...")
		print("")

		# Remove strongest feature from feature search space
		temp = col_names[strong_feature]
		data = data.drop([temp], axis=1)
		orig_data = orig_data.drop([temp], axis=1)
		col_names.remove(temp)
		num_features -= 1

	print("FEATURES:", features)

	temp_final = []
	final_a = []
	# This goes through looking for the top three strongest feature combination that provides the highest accuracy
	for n in range(0,total_runs):
		temp_final.append(features[n])
		best_total_accuracy = training(untouched_original,temp_final,0)
		final_a.append(best_total_accuracy)
		print("temp_final",temp_final, "accuracy",best_total_accuracy)
	max_accuracy = max(final_a)
	final_set = features[:final_a.index(max_accuracy)+1]

	print("Finished search!!! The best feature subset is ", final_set, "which has an accuracy of", max_accuracy*100, "%")
	return total_results

# Main loop, allows user to specify input text file, select search algorithm
# Then loop proceeds to normalize data and call the appropirate search algorithm
def main():
	print("Welcome to Jasmine's Feature Selection Algorithm\n")

	text_file = input("Type in the name of the file to test:")
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

	accuracy_percentage = accuracy*100
	print("Running nearest neighbor with all ", features_total-1, "features, using 'leaving-one-out' evaluation, I get an accuracy of ", accuracy_percentage, "%")

	print("Beginning search.")

	if algorithm == 1:
		forward_selection(df, features_total)
	elif algorithm == 2:
		backwards_elimination(df, features_total)
	elif algorithm == 3:
		jasmine_search_algorithm(df, features_total)
	else:
		print("Invalid")

	end = time.time()
	print("Time lapsed (seconds): ", end-start)

# Runs main
main()

