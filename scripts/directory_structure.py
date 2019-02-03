# coding: utf-8
#File to specify directory structure for project and to specify other methods
#related to reading and writin directories and removing file extensions.

import os 
import re 
  

def chooseDirectoryFromRoot(directory):	
	'''
	take directory to go to from the root directory of project 

	Args:
		directory (str): name of directory user wants to go to

	'''
	#go to root directory of project from current directory
	os.chdir("..")

	#go to directory specified
	os.chdir(directory)

def filesInDirectory(extension, directory):
	'''
	return the list of files in the directory with a specific extension

	Args:
		extension (str): file type to get

		directory (str): path of where files are present

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

def getWriteDirectory(directory_name, subdirectory_name):	
	'''
	get path of directory name specified where information needs
	to be written to (subdirectory specification is optional)
	
	Args:
		directory_name (str): name of directory to read from

		subdirectory (str): subdirectory of directory specified

	Returns:
		wr_dir (str): path of directory to write data to
	'''

	if subdirectory_name == None:
		wr_dir = os.getcwd() + '/../../' + directory_name
	else:
		if subdirectory_name == '/':
			wr_dir = os.getcwd() + '/../../' + directory_name + '/' + '_'
		else:	
			wr_dir = os.getcwd() + '/../../' + directory_name + '/' + subdirectory_name

	#if dir does not exist make new one
	if not os.path.exists(wr_dir):
		os.mkdir(wr_dir)
		return wr_dir
	else:    
		#return directory specified
		return wr_dir

def getReadDirectory(directory_name):	
	'''
	get the path of passed in directory name from root of project

	Args:
		directory_name (str): name of directory to read from

	Returns:
		rd_dir (str): path of directory to read data from
	'''

	rd_dir = os.getcwd() + '/../' + directory_name
	return rd_dir

def extractNumFromFile(file_name): 
	'''
	get maximum number from file name passed in

	Args;
		file_name (str): string to extract number from

	Returns:
		(str): max number from file name 
	'''
	numbers = re.findall('\d+',file_name) 
	numbers = map(int,numbers) 
	return str(numbers[0])
