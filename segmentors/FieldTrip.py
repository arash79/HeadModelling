import matlab.engine
import nibabel as nib
import numpy as np
import os
from rich.console import Console
import yaml

console = Console()
# loading the configuration file
with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config_file = yaml.safe_load(file)


def FieldTrip_Segmentation(nifti_file_path):

    """
    This function uses FieldTrip toolbox to segment MRI images into gray matter,
    white matter, CSF, skull and soft tissue.

    :param nifti_file_path: path to the nifti file
    """

    config = {'brainsmooth': 100,
              'scalpsmooth': 100,
              'skullsmooth': 100,
              'brainthreshold': 0.5,
              'scalpthreshold': 100,
              'skullthreshold': 1000,
              'downsample': 1,
              'output': {'tpm'},
              'spmmethod': 'mars'}

    console.log('[red underline]starting matlab engine...')  
    engine = matlab.engine.start_matlab()  # starting matlab engine

    engine.eval('warning off', nargout=0)  # turn off warnings

    # adding FieldTrip to path
    FieldTrip_PATH = engine.genpath(config_file['PATHs']['FieldTrip_PATH'])

    console.log('[bold blue]adding FieldTrip to path...')
    engine.addpath(FieldTrip_PATH)

    console.log('[bold blue]FieldTrip added to path...')
    _ = engine.ft_defaults

    console.log('[bold blue]reading MRI data...')
    file_name = nifti_file_path.split('\\')[-1][:-4]
    affine_transform = nib.load(nifti_file_path).affine
    MRI = engine.ft_read_mri(nifti_file_path)  # reading the MRI data using FieldTrip
    MRI['coordsys'] = 'ras'

    console.log('[bold blue]segmenting tissues...')
    segmented = engine.ft_volumesegment(config, MRI)  # segmenting the MRI data using FieldTrip 

    console.log('[bold blue]Finalizing results...')
    gray_matter = np.array(segmented['gray'])
    white_matter = np.array(segmented['white'])
    CSF = np.array(segmented['csf'])
    skull = np.array(segmented['bone'])
    soft_tissue = np.array(segmented['softtissue'])

    engine.quit()

    console.log('[bold blue]saving results...')

    GM_nifti = nib.Nifti1Image(gray_matter, affine_transform)
    nib.save(GM_nifti, os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                    "GM_{}.nii".format(file_name)))

    WM_nifti = nib.Nifti1Image(white_matter, affine_transform)
    nib.save(WM_nifti, os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                    "WM_{}.nii".format(file_name)))

    CSF_nifti = nib.Nifti1Image(CSF, affine_transform)
    nib.save(CSF_nifti, os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                     "CSF_{}.nii".format(file_name)))

    SKULL_nifti = nib.Nifti1Image(skull, affine_transform)
    nib.save(SKULL_nifti, os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                       "SKULL_{}.nii".format(file_name)))

    ST_nifti = nib.Nifti1Image(soft_tissue, affine_transform)
    nib.save(ST_nifti, os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                    "SOFT_TISSUE_{}.nii".format(file_name)))

    final_results = {'GM': gray_matter,
                     'WM': white_matter,
                     'CSF': CSF,
                     'SKULL': skull,
                     'SOFT_TISSUE': soft_tissue}

    return final_results
