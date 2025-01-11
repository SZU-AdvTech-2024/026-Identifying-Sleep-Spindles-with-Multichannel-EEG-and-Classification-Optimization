import os
import pandas as pd
import mne
from Filter_based_and_thresholding import Filter_based_and_thresholding
# os.chdir('~\\training set')
raw_file = 'DatabaseSpindles/excerpt1.edf'
# a_file = 'suj8_d2final_annotations.txt'
# # get the annotations if you have, generating a data frame that contain three columns: Onset, Duration, Annotation
# annotations = pd.read_csv(a_file)
# use MNE-python to read preprocessed EEG data. A raw object must be read before the later steps
raw = mne.io.read_raw_edf(raw_file,preload=True)
raw.drop_channels(['O1-A1', 'EMG1', 'ECG', 'NAF1', 'VAB', 'VTH', 'FP1-A1', 'EOG2-A1', 'EOG1-A1', 'POS', 'PHONO', 'SAO2', 'PR', 'PULSE', 'hypnogram', 'event_pneumo', 'hypnogram_aut', 'event_pneumo_aut', 'event_neuro'])
# initialize the object function
a=Filter_based_and_thresholding()
# step one: check the data online
a.get_raw(raw)
# # step two (optional if you want to output the probability of segmented EEG signals, in which whether it contains a spindle)
a.get_epochs()
# # step three: read the annotation data frame to the object for later purpose
a.get_annotation(annotations)
# step four (main step): use Filter-based and thresholding to find onsets and durations of sleep spindles
# the only input would be the signal itself and the lower/higher thresholds, no other information added
a.find_onset_duration(0.48,3.48)
# step five (optional): exclude sleep spindles found outside sleep stage 2
# a.sleep_stage_check()
# step six (optional): output probabilities of segmented signals of whether they contain a spindle
# a.predict_proba()
# step seven: generate true labels for cross validation or optimization the hyper-parameters
a.mauanl_label()
