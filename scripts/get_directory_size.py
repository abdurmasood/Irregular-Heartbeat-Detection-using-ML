#File used to estimate size of directory according to .png images it 
#already has. 

import signal_info
import directory_structure
import os

EXPECTED_SIZE_OF_DIR = 0

def getFileSize(file_path):
    return os.path.getsize(file_path)

if __name__ == '__main__':
    #path where beat images are located
    beats_path = directory_structure.getWriteDirectory('beat_write_dir')

    #path where signal files are located
    signal_path = directory_structure.getReadDirectory('mit-bih_waveform') 

    #get all png file names and save in a list
    beat_image_names = directory_structure.filesInDirectory('.png', beats_path)
    
    for image in beat_image_names:
        patient_id = signal_info.getNumbersFromString(image)[0]
        
        #find all available qrs locations of current patient id
        qrs_count = len(signal_info.getQRSLocations(signal_path + '\\' + patient_id))

        #get file size of produced image
        current_file_size = getFileSize(beats_path + '\\' + image)

        EXPECTED_SIZE_OF_DIR += (current_file_size*qrs_count) 


    print "The expectied size of the images all together is " + EXPECTED_SIZE_OF_DIR