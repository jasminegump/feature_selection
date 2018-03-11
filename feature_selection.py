import re
import pandas as pd
import math
import matplotlib.pyplot as plt

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
	print(df.c)

#def euclidean_dist(pt1, pt2):

def plot_save_fig(fig_num, plot_obj):
	fig = plt.figure()
	fig.show()
	plot_obj.plot()
	plt.savefig("fig"+str(fig_num)+".png")



text_file = 'CS205_SMALLtestdata__70.txt'
input_list = read_file(text_file)
list_to_pandas(input_list)