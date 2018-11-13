import pandas as pd
import os 
import matplotlib.pyplot as plt
import wfdb
import read_dataset

def takeAllInputs():
	'''
	function which takes input from user to display the waveform of the file selected 
	
	Returns:
			file_number (str): the number of file that needs to be displayed

			sample_from (int): start index of sample

			sample_to (int): end index of sample
	'''
	try:
		file_number = raw_input("\nWhat file do want to display \n")
		sample_from = raw_input("\nWhere do you want the sample to start \n")
		sample_to = raw_input("\nWhere do you want the sample to end \n")
		
		return str(file_number), int(sample_from), int(sample_to)
	except:
		print "Error please try again (Check if the name you're entering is in the database)"


if __name__ == '__main__':

	#find directory where data is
	read_dataset.chooseDirectoryFromRoot('mit-bih_waveform')

	#take user inputs
	file_number, sample_from, sample_to = takeAllInputs()

	#make record for file specified by user
	record = wfdb.rdrecord(file_number, sampfrom=sample_from, sampto=sample_to)

	#plot waveform
	wfdb.plot_wfdb(record=record, title='Record ' + file_number + ' from MIT-BIH Arrhythmia Database')

	display(record.__dict__)



