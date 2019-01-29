# script to generate data labels for images produced by script 
# extract_heartbeats

import pandas as pd
import directory_structure
import csv
import natsort  # module used to sort file names

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

		anns_df.to_pickle(pickle_ann_dir + '/' + directory_structure.removeFileExtension(annotation_file) + '.pkl')