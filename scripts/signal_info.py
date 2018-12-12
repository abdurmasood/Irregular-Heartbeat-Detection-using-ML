import pandas as pd
import os 
import wfdb
from wfdb import processing
import matplotlib.pyplot as plt
from PIL import Image
import directory_structure
import re

#number of heartbeats to extract
NUM_HEARTBEATS_TO_EXTRACT = 1
BEAT_START_OFFSET = 70
BEAT_END_OFFSET = 70

def takeAllInputs():
	'''
	function which takes input from user to display the waveform of the file selected 
	
	Returns:
		file_path (str): the location of file that needs to be displayed

		sample_from (int): start index of sample

		sample_to (int): end index of sample
	'''
	try:
		file_path = raw_input("\nWhat file do want to display \n")
		sample_from = raw_input("\nWhere do you want the sample to start \n")
		sample_to = raw_input("\nWhere do you want the sample to end \n")
		
		return str(file_path), int(sample_from), int(sample_to)
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

def getSignalInfo(file_path, sample_from, sample_to):
	'''
	Reads the signal and its fields

	Args:	
			file_path (str): the location of file whose beat needs to be displayed

			beat_start (int): start index of beat

			beat_end (int): end index of beat
	'''
	signal, fields = wfdb.rdsamp(file_path, sampfrom=sample_from, sampto=sample_to, channels=[0])
	return signal, fields

def writeSingleBeat(file_path, beat_start, beat_end, beat_number):
	'''
	Plots the single beat of a signal

	Args:	
			file_path (str): the location of file whose beat needs to be displayed

			beat_start (int): start index of beat

			beat_end (int): end index of beat

			beat_number (int): the index of what beat is currently being plotted
	
			wr_dir (str): directory to where beat needs to be written
	'''

	#save directory where beats need to be written
	beat_wr_dir = directory_structure.getWriteDirectory('beat_write_dir')

	#get signal and fields of specified file_path
	signal, fields = getSignalInfo(file_path, beat_start, beat_end)

	#plot beat
	plotItem(signal, beat_number, beat_wr_dir, file_path)

def plotItem(signal, beat_number, wr_dir, file_path):
	'''
	Plots and saves signal passed in current directory

	Args:	
			signal (list): list of intensity values of signal to be plotted

			beat_number (int): the index of what beat is currently being plotted

			wr_dir (str): directory to where beat needs to be written

			file_path (str): the location of file to plot
	'''
	file_number = (getNumbersFromString(file_path))[0]

	#plot color signal and save
	plt.plot(signal)
	plt.axis('off')
	plt.savefig(wr_dir + '\\image_' + file_number + '_' + str(beat_number), dpi=250)
	
	#convert grayscale and overwrite
	img = Image.open(wr_dir + '\\image_' + file_number + '_' + str(beat_number) + '.png').convert('LA')
	img.save(wr_dir + '\\image_' + file_number + '_' + str(beat_number) + '.png')

	#clear plot before next plot
	plt.clf()

'''
Takes a string and gets all the ints from that string

Args:
	string (str): string to find numbers from

Returns:
	(list): list of all number in the string 
'''
def getNumbersFromString(string):
	return (re.findall(r'\d+', string))


def plotSignal(file_path, sample_from, sample_to):
	'''
	plots the entire signal of specified file number
	
	Args:
		file_path (str): the location of file that needs to be displayed

		sample_from (int): start index of sample

		sample_to (int): end index of sample
	'''
	#make signal for file specified by user
	signal, fields = getSignalInfo(file_path, sample_from, sample_to)
	
	xqrs = getXQRS(signal, fields)

	#plot waveforms
	wfdb.plot_items(signal=signal, ann_samp=[xqrs.qrs_inds], title='Signal ' + file_path + ' from MIT-BIH Arrhythmia Database')
	
def getXQRS(signal, fields):
	'''
	The qrs detector class for the xqrs algorithm

	Args:
			signal (list): y values of signal samples
			fields (list): properties of signal

	Returns:
			xqrs (XQRS object): used to plot signals 
	'''
	#QRS Detection 
	xqrs = processing.XQRS(sig=signal[:,0], fs=fields['fs'])
	xqrs.detect(verbose=True)
	return xqrs

def getQRSLocations(file_path):
	'''
	get numpy list of QRS Locations

	Returns:
			qrs_locs (numpy list): list of QRS locations in the signal
	
	'''
	record = wfdb.rdrecord(file_path, channels=[0])
	qrs_locs = processing.gqrs_detect(record.p_signal[:,0], fs=record.fs)
	return qrs_locs


'''
finds qrs complexes in specified patient file

Args:
	file_path (str): path of where patient data is present
'''
def extractBeatsFromPatient(file_path):

	#get list of locations where QRS Complex happens
	qrs_locs = getQRSLocations(file_path)
	print "number of qrs locs of patient " + getNumbersFromString(file_path)[0] + " is " + str(len(qrs_locs))

	#uncomment to extract all heartbeats
	#NUM_HEARTBEATS_TO_EXTRACT = len(qrs_locs)

	#save directory where beats need to be written
	beat_wr_dir = directory_structure.getWriteDirectory('beat_write_dir')

	#plot and save the beats in the range selected
	for beat_number in range(NUM_HEARTBEATS_TO_EXTRACT):
		beat_start = qrs_locs[beat_number] - BEAT_START_OFFSET
		beat_end = qrs_locs[beat_number+1] - BEAT_END_OFFSET
		writeSingleBeat(file_path, beat_start, beat_end, beat_number)	
