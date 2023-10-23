# BrainSegmentation

This is a scientific in-progress work in collaboration with Tampere University, Finland, under the supervision of Dr. Samavaki & Prof. S. Pursiainen.

### Note: It is recommended to use this project in a virtual environment.

A full guide on how to set up a virtual environment in Python is available at:

https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/

This project relies heavily on some third-party libraries. In order to use these files you have to install those libraries first. You can install the mentioned libraries by entering the following command in your terminal (after activating the virtual environment):

```python
pip install -r requirements.txt
```

to have a hands-on experience with the current module, you need to make sure to have Matlab on your system.


# Algorithm Overview 

The Frangi filter is first applied to the MRI data slice-by-slice, and then the results are aggregated to produce the final arterial model. In summary, this process is executed through the following three sequential steps:

1. Frangi’s algorithm is applied to both the preprocessed INV2 and T1w slices of the dataset separately. Here the scikit-Image package’s implementation of the Frangi method is used. The set of parameters remains the same for each slice.


2. After applying the filter to a specific slice of the INV2 and T1w data, we created a mask by superposing these two layers in an element-wise manner. The mask was binarized using a threshold level; every element with a value less than this threshold was set to zero and, otherwise, to one.


3. The segmented cerebral vessels were obtained by iterating the previous steps through an axis of the MRI image and aggregating the results. In order to reduce noise, aggregation is performed separately for sagittal, axial, and coronal slices using the following scoring scheme: if a voxel is detected as a vessel in two or three of the results, it is considered a vessel in the final vessel mask; otherwise, it is neglected.


The codebase has been meticulously annotated and documented and for those seeking a deeper understanding of the algorithm and its practical implementation, additional insights, and details can be readily located within the code itself.


# Usage Guide

Initialize the configuration file, config.yaml, by adapting the paths and parameters to align with your specific requirements. You can discover the appropriate parameters through a process of trial and error, experimenting a few times to determine the best fit. The paths are clear to understand but a minor description about the ones that are starting with the prefix `EXTRACTED` is needed. These are the paths to the skull-stripped versions of the MRI data that we have. I have used the Brain Extraction Toolkit (BET) of the FSL software to perform the skull-stripping process. The link to their wiki page is provided below:

https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

* Remark: Keep in mind that the quality of your final model relies heavily on the quality of the files obtained from the skull-stripping process. So be careful about the parameters that you use for the stripping process and do not blindly use the default parameters. It is recommended to manually inspect the quality of the results before passing them to the algorithm.


The parameters and their descriptions are as follows:


- [x] **region_growing_threshold**:
The region-growing algorithm aims to rectify potential misses within the original algorithm. It takes a voxel as a seed and tries to increase its domain by finding similar adjacent voxels to itself. The similarity between the voxels is defined in terms of intensity. During this process, the similarity or closeness of the intensity of adjacent voxels needs to be defined. In this work, two adjacent voxels are deemed to be similar in nature if their intensities are equal within a tolerance. This tolerance is a floating number and is defined using this parameter.


- [x] **drop_intensively** & **connected_components_minimum_length**:
Following the extraction of arteries through the algorithm, despite the application of existing noise reduction techniques to the model, it is still possible that a notable amount of noise remains perceptible. To address this issue a mechanism is designed to remove or keep these voxels according to the user's choice. The mechanism works as follows: It searches through the entire 3-dimensional resulting array and finds all of its connected components. The size of these components can vary but are at least 1. If the parameter **drop_intensively** is set to be True then the connected components which are of size less than **connected_components_minimum_length** will be dropped in the final result. As stated earlier this step is optional and can be determined by a boolean value.


- [x] **dismiss_soft_tissue_portion**:
As said before, there can be some noise in the resulting model. Another mechanism that is considered to reduce the noise and increase the accuracy of the algorithm in classifying voxels as arteries, is to use the FieldTrip toolbox segmentation technique to mask out the unwanted parts from the final vessel model. We first segment both INV2 and T1w modalities to five different tissues including White Matter, Grey Matter, CSF, Skull, and Soft tissue using the FieldTrip toolbox. Then we save the resulting segmentation in separate files in the ```results\FieldTrip``` directory as nifti files. According to my experience the most probable segments to be misclassified as arteries are the soft tissue and skull. I use these segments and add them together in an element-wise manner to create a mask that consists of the voxels that are determined to be soft tissue or skull in either of the modalities. The noise reduction algorithm tries to remove the voxels that are both present in the tissue mask and the segmented artery model, hoping to result in a better arterial model. My experiments showed that the density of the tissue mask tends to be higher around the center of the head (our MRI data is in-vivo hence this issue is natural because of the presence of some organs like the throat and muscles). Having this in mind one can safely conclude that removing this portion of the tissue mask from the arterial model can lead to unwanted loss of arteries, especially the arteries that are around of brain stem. To fix this problem we set the values of the voxels around this region in the tissue mask to zero. The current parameter is used to determine this region. This parameter is a tuple of two integers like (a, b), signifying that voxels falling within the range of 1/a to 1/b of the tissue mask's height will be excluded (set to zero).

Once the parameters have been initialized and the file path values have been modified, you can initiate the algorithm by simply executing the following command:

`python main.py`

As attaining a satisfactory model may necessitate multiple iterations of the algorithm with different sets of parameters, and considering that FieldTrip segmentation can be quite time-intensive, you have the option to comment out the following two lines in the `main.py` module after the initial execution of the algorithm.

```python
_ = FieldTrip_Segmentation(config['PATHs']['T1w_PATH'])
_ = FieldTrip_Segmentation(config['PATHs']['INV2_PATH'])
```

This can expedite the trial-and-error process. This is due to the fact that these parameters do not exert any influence on the FieldTrip segmentation, and after the initial execution, the segmentation results are yielded. Thus, there is no need to recalculate them when refining your model.


# Future Work

Given the algorithm's high level of accuracy, it is well-suited for the task of labeling voxels from diverse MRI datasets, thus enabling the creation of a robust labeled dataset for utilization in a supervised machine-learning approach. This, in turn, can significantly enhance the accuracy of artery segmentation, particularly when employing techniques like Convolutional Neural Networks (CNNs).
