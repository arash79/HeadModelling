import yaml
import os
from segmentors import *
from STLs import *

# loading the configuration file
with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = yaml.safe_load(file)

os.makedirs(config['PATHs']['FieldTrip_RESULTS_PATH'], exist_ok=True)  # making sure that the FieldTrip_RESULTS_PATH exists
os.makedirs(config['PATHs']['RESULTS_PATH'], exist_ok=True)  # making sure that the RESULTS_PATH exists

if __name__ == '__main__':
    _ = FieldTrip_Segmentation(config['PATHs']['T1w_PATH'])  # segmentation of the T1w image
    _ = FieldTrip_Segmentation(config['PATHs']['INV2_PATH'])  # segmentation of the INV2 image
    segmentor = SegmentArteries()  
    result = segmentor.segment_arteries()  # segmentation of the arteries
    create_stl_file_from_array(given_array=result, 
                               output_file_name='arteries')  # creation of the STL file named arteries.stl
