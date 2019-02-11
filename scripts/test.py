import wfdb
from wfdb import processing
import directory_structure
import natsort
import signal_info

#find directory where data is
signal_dir = directory_structure.getReadDirectory('mit-bih_waveform')

#get all .hea and .dat files (respectively)
signal_files = directory_structure.filesInDirectory('.hea', signal_dir)

# sort file names in ascending order in list
signal_files = natsort.natsorted(signal_files)

for signal_file in signal_files:
    print(signal_file)
    file_dir = signal_dir + '/' + directory_structure.removeFileExtension(signal_file)

    # get annotation of signal
    ann = wfdb.rdann(file_dir, 'atr', return_label_elements=['symbol', 'description', 'label_store'] , summarize_labels=True)
    sig, fields = wfdb.rdsamp(file_dir, channels=[0], sampfrom=0, sampto=5000)

    if signal_file == '106.hea':
        print(ann.sample)
        print(ann.symbol)
    print() 

    # get signal
    # sig, fields = wfdb.rdsamp(file_dir, channels=[0], sampfrom=0, sampto=3000)
    # xqrs = processing.XQRS(sig=sig[:,0], fs=fields['fs'])
    # xqrs.detect()

    # wfdb.plot_items(signal=sig, ann_samp=[xqrs.qrs_inds])

    # print signal_info.getQRSLocations(signal_dir + '/' + directory_structure.removeFileExtension(signal_file))
    