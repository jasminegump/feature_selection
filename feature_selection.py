import re
import pandas as pd
import math
import matplotlib.pyplot as plt
import operator 
import numpy as np
import copy
import sys

# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

col_list = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10']

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
	df = pd.DataFrame(input_list, columns=['c','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])
	return df
	#print(df.c)

def euclidean_distance(x1, x2):
	return math.sqrt((x1-x2)**2)

def get_neighbors(train_set, test):
	distance = []
	for i in range(len(train_set)):
		x1 = train_set[i][1]
		x2 = test[1]
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

def training(df):
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

def main():
	df = read_input('CS205_SMALLtestdata__70.txt')
	training(df)


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

	





