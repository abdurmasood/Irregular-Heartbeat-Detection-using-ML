# interface for interacting with pickle data and generating pickle data
# run this script to generate pickle files of annotation dataframes 
# saved in specified directory

import pandas as pd
import directory_structure
import csv
import natsort  # module used to sort file names

def getAnnotationDataFrame(file_name):
	'''
	returns dataframe that corresponds to the important annotation data of
	specific pickle file

	Args:
		file_name (str): name of pickle file to read

	Returns:
		(dataframe): dataframe of annotation information
	'''
	
	ann_dir_pkl = directory_structure.getWriteDirectory('pickle_annotation_data') + '/' + directory_structure.removeFileExtension(file_name) + '.pkl'
	return pd.read_pickle(ann_dir_pkl)


if __name__ == '__main__':
	# get directory of annotations text files
	ann_dir = directory_structure.getReadDirectory('mit-bih_database')
	ann_files = directory_structure.filesInDirectory(".txt", ann_dir)

	# get dir where pickle data needs to be written
	pickle_ann_dir = directory_structure.getWriteDirectory('pickle_annotation_data')

	# sort file names in ascending order in list
	ann_files = natsort.natsorted(ann_files)

	for annotation_file in ann_files:
		# read annotation files where the delimiter is 1 or more spaces and store in data frame
		anns_df = pd.read_csv(ann_dir + '/' + annotation_file, quoting=csv.QUOTE_NONE, delimiter=r"\s+", encoding='utf-8')

		# overwrite original dataframe by including only relevant parameters
		anns_df_temp = {'Time': anns_df['Time'], 'Sample #': anns_df['Sample'], 'Type': anns_df['#']}
		anns_df = pd.DataFrame(data=anns_df_temp)

		anns_df.to_pickle(pickle_ann_dir + '/' + directory_structure.extractNumFromFile(annotation_file) + '.pkl')
