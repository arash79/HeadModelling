from multiprocessing.dummy import Pool
from warnings import filterwarnings
from skimage.filters import frangi
from skimage import measure
import nibabel as nib
import pandas as pd
import numpy as np
from typing import Tuple
import time
from rich.console import Console
import os
import yaml

filterwarnings('ignore')

console = Console()

with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config_file = yaml.safe_load(file)


class SegmentArteries:

    """
    This class is used to segment the arteries from the MRI images. It uses the Frangi 
    vesselness filtering to detect the vessels, and then it uses region growing algorithm
    to expand the vessels to compensate for possible mis-classifications.

    It also uses the FieldTrip segmentation results to mask out the soft tissue and skull 
    in order to reduce the noise in the final result.
    """

    def __init__(self):

        T1w = nib.load(config_file['PATHs']['EXTRACTED_T1w_PATH']).get_fdata()
        INV2 = nib.load(config_file['PATHs']['EXTRACTED_INV2_PATH']).get_fdata()

        self.affine_transform = nib.load(config_file['PATHs']['EXTRACTED_T1w_PATH']).affine

        # normalizing the data
        self.T1w = (T1w - np.amin(T1w)) / (np.amax(T1w) - np.amin(T1w))
        self.INV2 = (INV2 - np.amin(INV2)) / (np.amax(INV2) - np.amin(INV2))

        self.shape = self.T1w.shape

        # this list is used in the region growing algorithm to keep track of visited vortices
        self.traversed = []  

    @staticmethod
    def __filter_connected_components(array: np.array, 
                                      length_threshold: int) -> np.array:

        """
        This method filters out connected components of any array which have size smaller
        than a given length threshold
        :param array: The given N-D array
        :param length_threshold: The desired minimum length
        :return: Filtered array
        """

        labeled_array = measure.label(array, connectivity=2)  # connectivity=2 means 8-connectivity
        properties = measure.regionprops(labeled_array)
        keep_mask = np.zeros_like(labeled_array, dtype=bool)

        for property_ in properties:  # looping over the connected components
            # finding the connected components with length smaller than the threshold
            if property_.major_axis_length < length_threshold:
                keep_mask[labeled_array == property_.label] = True

        filtered_array = array.copy()  # making a copy of the given array to avoid changing it
        filtered_array[keep_mask] = 0  # discarding the components with length less than the threshold

        return filtered_array

    def __check_dim(self, coordinate: Tuple) -> bool:

        """
        This method checks whether the given coordinate lies in the boundary of the 
        current array or it exceeds its boundaries. This method is used in the region 
        growing algorithm.

        :param coordinate: given coordinate system
        :return: True if the given coordinate is inside the plane otherwise it returns False
        """

        excess_x = coordinate[0] < 0 or coordinate[0] >= self.shape[0] - 1  # -1 because of the zero-based indexing
        excess_y = coordinate[1] < 0 or coordinate[1] >= self.shape[1] - 1  # -1 because of the zero-based indexing

        return not (excess_x or excess_y)

    def __region_growing(self, 
                         given_seed: Tuple, 
                         vessel_mask: np.array, 
                         masked_cross_section: np.array,
                         threshold: float, 
                         voxel_value: float) -> None:

        """
        This method expands the given seed region by checking that whether its neighbors have an intensity value within
        the tolerance range of the given seed intensity. If True it announces them as vessels otherwise it let them be.

        :param given_seed: The seed which was provided using the Frangi vesselness filtering
        :param vessel_mask: The whole cross-section consisting of different seeds obtained by the Frangi method
        :param masked_cross_section: The weighted average mask of T1w and INV2 cross-sections
        :param threshold: The tolerance range of voxels intensities similarities
        :param voxel_value: The minimum intensity value to detect a voxel as a vessel, It is used to reduce noise
        :return: None, It effects directly on to the given vessel mask
        """

        stack = [given_seed]  # initializing the stack with the given seed

        while len(stack):  # while the stack is not empty

            current_seed = stack.pop()  # popping the last element of the stack

            # if the current seed has been visited before, skip it
            if current_seed in self.traversed:  
                continue
            
            # if the current seed is not as bright as a vessel might be, mark it as False, then skip it
            if masked_cross_section[current_seed] < voxel_value:  
                vessel_mask[current_seed] = np.False_  
                continue
            
            # extracting the coordinates of the current seed
            current_x, current_y = current_seed[0], current_seed[1] 

            adjacent_voxels = []  # initializing the list of adjacent voxels
            possible_moves = [-1, 0, 1]  # possible moves in the x and y directions

            for i in possible_moves:  # looping over the possible moves
                for j in possible_moves:
                    if i != 0 or j != 0:  # if the move is not a stay
                        neighbor = (current_x + i, current_y + j)  # finding the neighbor
                        if not vessel_mask[neighbor]:  # if the neighbor is not already a vessel
                            adjacent_voxels.append(neighbor)

            self.traversed.append(current_seed)  # marking the current seed as visited

            for neighbor in adjacent_voxels:  # looping over the adjacent voxels
                # if the neighbor is similar to the current seed in terms of intensity
                if np.isclose(masked_cross_section[current_seed], masked_cross_section[neighbor], rtol=threshold):
                    vessel_mask[neighbor] = np.True_  # mark the neighbor as a vessel
                    # if the neighbor is not already in the stack and yet it is inside the cross-section
                    if neighbor not in stack and self.__check_dim(neighbor):
                        stack.append(neighbor)

    def __segment_arteries(self) -> np.array:

        """
        This method generates the vessel mask using the Frangi vesselness filtering. 
        Since it generates noisy results it is protected and will be called from another
        method which is coded below. This method calls uses multiprocessing to increase
        the calculations speed.

        :return: vessel mask (which by the way can be noisy)
        """
        # initializing the axes
        first_axis, second_axis, third_axis = np.zeros(self.shape), np.zeros(self.shape), np.zeros(self.shape)  
        # extracting the lengths of the axes
        first_axis_length, second_axis_length, third_axis_length = self.shape[0], self.shape[1], self.shape[2]  

        def frangi_filter(T1w_cross_section: np.array, 
                          INV2_cross_section: np.array, 
                          axis: str,
                          layer_index: int) -> None:
            
            """
            This function is used to apply the Frangi vesselness filtering on the given 
            cross-sections. It also uses the region growing algorithm to expand the filtered
            voxels.

            :param T1w_cross_section: T1w cross-section
            :param INV2_cross_section: INV2 cross-section
            :param axis: The axis along which the cross-section is taken
            :param layer_index: The index of the cross-section along the given axis
            """

            # applying the Frangi vesselness filtering on the T1w cross-section
            T1w_result = frangi(T1w_cross_section, 
                                sigmas=np.arange(.5, 1.5, .25), 
                                alpha=0.9, 
                                beta=0.9,
                                gamma=40, 
                                black_ridges=False) 
            # applying the Frangi vesselness filtering on the INV2 cross-section
            INV2_result = frangi(INV2_cross_section, 
                                 sigmas=np.arange(.5, 1.5, .25), 
                                 alpha=0.9, 
                                 beta=0.9,
                                 gamma=40, 
                                 black_ridges=False)

            # since the arteries are among the brightest parts of the image, we can use 
            # the 99th percentile of the array to filter the brightest parts which we 
            # are certain that the arteries are a part of. This may also result in the 
            # presence of some unwanted parts too because it solely filters the voxels 
            # based on their brightness level. The value 99 is obtained heuristically
            # and can be changed if deemed necessary.
            try:
                T1w_threshold = np.percentile(T1w_result[np.nonzero(T1w_result)], 99)
            except IndexError:  # if the array is empty or has no nonzero elements
                T1w_threshold = 0

            # descriptions are the same as above
            try:
                INV2_threshold = np.percentile(INV2_result[np.nonzero(INV2_result)], 97)
            except IndexError:
                INV2_threshold = 0

            # marking the voxels with intensity higher than the threshold as True, 
            # getting rid of the rest of the voxels that are more likely to not be an
            # artery according to their intensity
            T1w_result = T1w_result > T1w_threshold  
            INV2_result = INV2_result > INV2_threshold
            
            # creating a dataframe to calculate the eigenvalues and use them as weights
            layer_dataframe = pd.DataFrame({'T1w': T1w_cross_section.ravel(),
                                            'INV2': INV2_cross_section.ravel()})
            correlation_matrix = layer_dataframe.corr()  # calculating the correlation matrix

            try:  # calculating the eigenvalues of the correlation matrix
                eigen_values = np.linalg.eig(correlation_matrix)[0]
                T1w_weight, INV2_weight = eigen_values[0], eigen_values[1]
            except np.linalg.LinAlgError:  # if the correlation matrix is singular
                T1w_weight, INV2_weight = 1, 1

            T1w_weight, INV2_weight = INV2_weight, T1w_weight  # swapping the weights

            # calculating the numerator of the mask
            numerator = (T1w_weight * T1w_cross_section) + (INV2_weight * INV2_cross_section) 
            # calculating the denominator of the mask 
            denominator = (T1w_weight + INV2_weight)
            masked_cross_section = numerator / denominator  # calculating the weighted average mask

            # complete description of the following lines are provided above
            try:
                voxel_threshold = np.percentile(masked_cross_section[np.nonzero(masked_cross_section)], 90)
            except IndexError:
                voxel_threshold = 0

            # combining the results of the Frangi filtering
            masked_frangi_results = T1w_result | INV2_result  

            # simply extracting the brightest parts of the weighted average mask,
            # although it is named vessel_mask it is just a mask of the brightest 
            # structures of the current cross-section obtained by naive thresholding.
            # however all the arteries are highly probable to be present in this mask
            vessel_mask = (masked_cross_section > voxel_threshold)

            # combining the results of the Frangi filtering and naive thresholding of 
            # the weighted average mask
            filtered_vessels = masked_frangi_results & vessel_mask 

            nonzero_elements = np.nonzero(filtered_vessels)  # extracting the nonzero elements
            vessel_seeds = list(zip(nonzero_elements[0], nonzero_elements[1]))  # creating the seeds

            for seed in vessel_seeds:  # looping over the seeds
                # applying the region growing algorithm on the seeds
                self.__region_growing(given_seed=seed,
                                      vessel_mask=filtered_vessels,
                                      masked_cross_section=masked_cross_section,
                                      threshold=config_file['PARAMETERs']['region_growing_threshold'],
                                      voxel_value=voxel_threshold)

            if axis == 'first':  # if the cross-section is taken along the first axis (sagittal)
                # assigning the filtered vessels to the first axis array at the given index
                first_axis[layer_index, :, :] = filtered_vessels
            if axis == 'second':  # if the cross-section is taken along the second axis (axial)
                # assigning the filtered vessels to the second axis array at the given index
                second_axis[:, layer_index, :] = filtered_vessels
            if axis == 'third':  # if the cross-section is taken along the third axis (coronal)
                # assigning the filtered vessels to the third axis array at the given index
                third_axis[:, :, layer_index] = filtered_vessels

        # creating a list of parameters to pass to the multiprocessing pool
        parameters_along_first_axis = [(self.T1w[i, :, :], self.INV2[i, :, :], 'first', i) for i in range(first_axis_length)]
        parameters_along_second_axis = [(self.T1w[:, i, :], self.INV2[:, i, :], 'second', i) for i in range(second_axis_length)]
        parameters_along_third_axis = [(self.T1w[:, :, i], self.INV2[:, :, i], 'third', i) for i in range(third_axis_length)]

        time_of_start = time.time()
        console.log('[bold green]applying vesselness filtering...')
        # applying the Frangi filtering on the cross-sections along the first axis
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_first_axis)
        console.log('[bold green]step 1 of 3 completed.')
        # applying the Frangi filtering on the cross-sections along the second axis
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_second_axis)
        console.log('[bold green]step 2 of 3 completed.')
        # applying the Frangi filtering on the cross-sections along the third axis
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_third_axis)
        console.log('[bold green]step 3 of 3 completed.')

        console.log('[bold green]total elapsed time {} seconds'.format(round(time.time() - time_of_start)))

        # merging the results of the Frangi filtering along the three axes and applying
        # a scoring scheme: if a voxel is detected as a vessel in two or three of the 
        # results, it is considered a vessel in the final vessel mask; otherwise, it is
        # neglected.
        merged_result = ((first_axis.astype(int) + second_axis.astype(int) + third_axis.astype(int)) >= 2)

        return merged_result

    def segment_arteries(self) -> np.array:

        """
        This method calls the private method of above. It requires the user to perform the FieldTrip segmentation on
        both T1w and INV2 modalities in advance, because it uses some of its results to mask out unwanted noises in the
        raw vessel mask obtained by the above method.

        :return: vessel mask
        """

        # creating the paths to the FieldTrip segmentation results
        T1w_soft_tissue_path = os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                            'SOFT_TISSUE_T1w.nii')
        T1w_skull_path = os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                      'SKULL_T1w.nii')
        INV2_soft_tissue_path = os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                             'SOFT_TISSUE_INV2.nii')
        INV2_skull_path = os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                       'SKULL_INV2.nii')

        try:  # loading the FieldTrip segmentation results
            T1w_soft_tissue = nib.load(T1w_soft_tissue_path).get_fdata()
            T1w_skull = nib.load(T1w_skull_path).get_fdata()
            INV2_soft_tissue = nib.load(INV2_soft_tissue_path).get_fdata()
            INV2_skull = nib.load(INV2_skull_path).get_fdata()
        except FileNotFoundError:  # if the FieldTrip segmentation results are not found
            console.log('[bold red]The required files for cleaning the vessel mask are not found. Please run the FieldTrip segmentation algorithm first.')
            exit()

        segmented_vessels = self.__segment_arteries()  # obtaining the raw vessel mask

        # combining the FieldTrip results to mask out the soft tissue and skull from 
        # the vessel mask
        extra_voxels = T1w_soft_tissue + INV2_soft_tissue + T1w_skull + INV2_skull 

        mask = np.zeros_like(extra_voxels, dtype=bool)  # initializing the mask
        # marking the voxels that are not soft tissue or skull as True
        mask[extra_voxels == 0] = True  

        # calculating the boundaries of the spared area as explained in readme file
        dismissal_portion = eval(config_file['PARAMETERs']['dismiss_soft_tissue_portion'])
        lower_boundary, upper_boundary = max(dismissal_portion), min(dismissal_portion)
        spared_area_boundaries = (self.shape[2] // lower_boundary, self.shape[2] // upper_boundary)

        # marking the spared area as True, details are explained in readme file
        mask[:, :, spared_area_boundaries[0]: spared_area_boundaries[1]] = True

        # applying the mask on the raw vessel mask
        cleaned_vessels = np.where(mask, segmented_vessels, 0)

        # filtering the connected components of the vessel mask as explained in readme file
        if config_file['PARAMETERs']['drop_intensively'] is True:
            cleaned_vessels = SegmentArteries.__filter_connected_components(array=cleaned_vessels,
                                                                            length_threshold=config_file['PARAMETERs']['connected_components_minimum_length'])

        # saving the cleaned vessel mask as a nifti file as explained in readme file
        if config_file['PARAMETERs']['save_as_nifti'] is True:
            vessel_nifti = nib.Nifti1Image(cleaned_vessels, self.affine_transform)
            nib.save(vessel_nifti, os.path.join(config_file['PATHs']['RESULTS_PATH'], "vessel_mask.nii"))

        return cleaned_vessels
