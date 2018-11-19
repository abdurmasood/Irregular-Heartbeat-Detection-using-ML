import pandas as pd
import os 
import matplotlib.pyplot as plt
import wfdb
from wfdb import processing
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

def calculateHeartRate(sample_to, xqrs, fs):
	'''
	Calculates the heart rate of signal uptil the point specified

	Args:
		sample_to (int): end sample of signal

		xqrs (signal): used for qrs detection

		fs (int): frequency of signal

	Returns:
		heart_rate (int): heart beat of person
	'''
	heart_rate_list = processing.compute_hr(sample_to, xqrs.qrs_inds, fs)
	heart_rate = heart_rate_list[-1]
	return heart_rate

if __name__ == '__main__':

	#find directory where data is
	read_dataset.chooseDirectoryFromRoot('mit-bih_waveform')

	#take user inputs
	file_number, sample_from, sample_to = takeAllInputs()

	#make signal for file specified by user
	signal, fields = wfdb.rdsamp(file_number, sampfrom=sample_from, sampto=sample_to, channels=[0])

	#normalize signal
	signal = processing.normalize_bound(signal, 0, 100)

	#QRS Detection 
	xqrs = processing.XQRS(sig=signal[:,0], fs=fields['fs'])
	xqrs.detect()

	#heart rate
	print calculateHeartRate(sample_to, xqrs, 360)

	#plot waveform
	wfdb.plot_items(signal=signal, ann_samp=[xqrs.qrs_inds], title='Signal ' + file_number + ' from MIT-BIH Arrhythmia Database')

