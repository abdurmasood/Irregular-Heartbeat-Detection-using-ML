import pandas as pd
import os 
import matplotlib.pyplot as plt


database_directory = "mit-bih_database"

def chooseDirectoryFromRoot(directory):	
	'''
	function which takes in the directory to go to from the root directory of project 

	Args:
		directory (str): name of directory user wants to go to

	'''
	#go to root directory of project from current directory
	os.chdir("..")

	#go to directory specified
	os.chdir(directory)

def filesInDirectory(extension):
	'''
	returns the list of files in the directory with a specific extension

	Args:
		extension (str): file type to get

	Returns:
		l: list of file names with extension (in current directory)
	'''
	l = []

	#get all file names in current directory
	file_names = os.listdir(os.getcwd())
	
	for file in file_names:
		if file.endswith(extension):
			l.append(file)

	return l

if __name__ == '__main__':

	#find directory where data is
	chooseDirectoryFromRoot(database_directory)

	#get all .txt and .csv files
	data_files = filesInDirectory(".csv")
	annotation_files = filesInDirectory(".txt")

	#read data of csv files iteratively
	# for file in database_files:
	# 	print pd.read_csv(file)
	# 	print ""
	# 	print ""
	
	#skip the name row and read all the rest data along with saving it in a dataframe
	first_csv_df = pd.read_csv(data_files[0], skiprows=1, names=['MLII', 'V1'])

	#skip the first two lines and save the rest data in the dataframe (use delimeter as space '    ')
	first_txt_df = pd.read_csv(annotation_files[0], skiprows=2, sep='    ', engine='python', names=['time', 'sample_no', 'type', 'sub', 'chan', 'num'])


	print first_csv_df.head(5)
	print ""
	print first_txt_df.head(5)
