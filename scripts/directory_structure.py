# coding: utf-8
#File to specify directory structure for project and to specify other methods
#related to reading and writin directories and removing file extensions.

import os 

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

def filesInDirectory(extension, directory):
	'''
	returns the list of files in the directory with a specific extension

	Args:
		extension (str): file type to get

	Returns:
		l (list): list of file names with extension (in current directory)
	'''
	l = []

	#get all file names in current directory
	file_names = os.listdir(directory)
	
	for file in file_names:
		if file.endswith(extension):
			l.append(file)

	return l

def removeFileExtension(file):
	'''
	remove extension of file passed in as a string

	Args:
		file (str): name of file with extension

	Returns:
		(str): name of file without extension
	'''
	return os.path.splitext(file)[0]

def getWriteDirectory(directory_name):	
	'''
	function which gets path of directory specified
	Args:
		directory_name (str): name of directory to read from

	Returns:
		wr_dir (str): path of directory to write data to
	'''

	wr_dir = os.getcwd() + '/../../' + directory_name

	#if dir does not exist make new one
	if not os.path.exists(wr_dir):
		os.mkdir(wr_dir)
		return wr_dir
	else:    
		#return directory specified
		return wr_dir

def getReadDirectory(directory_name):	
	'''
	function which gets the path of passed in directory name from root of project

	Args:
		directory_name (str): name of directory to read from

	Returns:
		rd_dir (str): path of directory to read data from
	'''

	rd_dir = os.getcwd() + '/../' + directory_name

	return rd_dir
