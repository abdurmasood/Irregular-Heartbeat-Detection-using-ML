import pandas as pd
import os 
import wfdb
import display_signal

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
		l (list): list of file names with extension (in current directory)
	'''
	l = []

	#get all file names in current directory
	file_names = os.listdir(os.getcwd())
	
	for file in file_names:
		if file.endswith(extension):
			l.append(file)

	return l

def removeFileExtension(file):
	'''
	remove extension of file passed in as a string

	Args:
			file (str): name of file
	'''
	return os.path.splitext(file)[0]

if __name__ == '__main__':

	#find directory where data is
	chooseDirectoryFromRoot('mit-bih_waveform')

	#get all .hea and .dat files (respectively)
	signals_files = filesInDirectory(".hea")
	dat_files = filesInDirectory(".dat")

	#extract and save beats from file provided
	for signal_file in signals_files:
		display_signal.extractBeatsFromPatient(removeFileExtension(signal_file))