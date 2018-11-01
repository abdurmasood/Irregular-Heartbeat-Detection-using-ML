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

def appendAllDataIntoOneDataFrame(file_names, rows_to_skip, delimeter, engine_name, data_frame_headings):
	'''
	returns the list of appended data from seperate files into a single dataframe

	Args:
		file_names (list): list of all the file names of a current extension in that directory
		
		rows_to_skip (int): rows to skip while reading from files in database directory (usually to skip name row)
		
		delimeter (str): used as a check to see if the reading is done on a .csv file or any other extension file
		
		engine_name (str): name of engine being used (in this case python)
		
		data_frame_headings (list): names of headings you want in the dataframe

	Returns:
		appended_data: dataframe with all the data of that specific extension appended together 
	'''	

	appended_data = pd.DataFrame()

	#read data of csv files iteratively
	for file in file_names:
		print file
		#check to see if reading csv or text files
		if delimeter == None:		
			appended_data = appended_data.append(pd.read_csv(file, skiprows=rows_to_skip, engine=engine_name, names=data_frame_headings))
		else:
			appended_data = appended_data.append(pd.read_csv(file, skiprows=rows_to_skip, sep=delimeter, engine=engine_name,  names=data_frame_headings))

	return appended_data

if __name__ == '__main__':

	#find directory where data is
	chooseDirectoryFromRoot(database_directory)

	#get all .csv and .txt files (respectively)
	data_files = filesInDirectory(".csv")
	annotation_files = filesInDirectory(".txt")

	signal_data_df = appendAllDataIntoOneDataFrame(data_files, 1, None, 'python', ['MLII', 'V1'])
	annotation_data_df = appendAllDataIntoOneDataFrame(annotation_files, 2 , '    ', 'python', ['time', 'sample_no', 'type', 'sub', 'chan', 'num'])
