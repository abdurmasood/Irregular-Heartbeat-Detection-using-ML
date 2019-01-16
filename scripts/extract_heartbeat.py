#File to save images of beats in a specified directory

import os 
import wfdb
import signal_info
import directory_structure


if __name__ == '__main__':

	#find directory where data is
	# directory_structure.chooseDirectoryFromRoot('mit-bih_waveform')

	signal_dir = directory_structure.getReadDirectory('mit-bih_waveform')

	#get all .hea and .dat files (respectively)
	signal_files = directory_structure.filesInDirectory(".hea", signal_dir)

	#extract and save beats from file provided
	for signal_file in signal_files:
		#uncomment to save images of beats
		signal_info.extractBeatsFromPatient(signal_dir + '\\' + directory_structure.removeFileExtension(signal_file))
