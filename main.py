import yaml
import os
from segmentors import *
from STLs import *


with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = yaml.safe_load(file)

os.makedirs(config['PATHs']['FieldTrip_RESULTS_PATH'], exist_ok=True)
os.makedirs(config['PATHs']['RESULTS_PATH'], exist_ok=True)

if __name__ == '__main__':
    _ = FieldTrip_Segmentation(config['PATHs']['T1w_PATH'])
    _ = FieldTrip_Segmentation(config['PATHs']['INV2_PATH'])
    segmentor = SegmentArteries()
    result = segmentor.segment_arteries()
    create_stl_file_from_array(given_array=result, 
                               output_file_name='arteries')
