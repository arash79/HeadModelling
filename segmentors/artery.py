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

    def __init__(self):

        T1w = nib.load(config_file['PATHs']['EXTRACTED_T1w_PATH']).get_fdata()
        INV2 = nib.load(config_file['PATHs']['EXTRACTED_INV2_PATH']).get_fdata()

        self.affine_transform = nib.load(config_file['PATHs']['EXTRACTED_T1w_PATH']).affine

        # normalizing the data
        self.T1w = (T1w - np.amin(T1w)) / (np.amax(T1w) - np.amin(T1w))
        self.INV2 = (INV2 - np.amin(INV2)) / (np.amax(INV2) - np.amin(INV2))

        self.shape = self.T1w.shape

        self.traversed = []

    @staticmethod
    def __filter_connected_components(array: np.array, 
                                      length_threshold: int) -> np.array:

        """
        This method filters out connected components of any array which have size smaller than a given length threshold
        :param array: The given N-D array
        :param length_threshold: The desired minimum length
        :return: Filtered array
        """

        labeled_array = measure.label(array, connectivity=2)  # connectivity=2 means 8-connectivity
        properties = measure.regionprops(labeled_array)
        keep_mask = np.zeros_like(labeled_array, dtype=bool)

        for property_ in properties:
            if property_.major_axis_length < length_threshold:
                keep_mask[labeled_array == property_.label] = True

        filtered_array = array.copy()
        filtered_array[keep_mask] = 0

        return filtered_array

    def __check_dim(self, coordinate: Tuple) -> bool:

        """
        This method checks whether the given coordinate lies in the boundary of the current array or it exceeds its
        boundaries. This method is used in the region growing algorithm.

        :param coordinate: given coordinate system
        :return: True if the given coordinate is inside the plane otherwise it returns False
        """

        excess_x = coordinate[0] < 0 or coordinate[0] >= self.shape[0] - 1
        excess_y = coordinate[1] < 0 or coordinate[1] >= self.shape[1] - 1

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

        stack = [given_seed]

        while len(stack):

            current_seed = stack.pop()

            if current_seed in self.traversed:
                continue

            if masked_cross_section[current_seed] < voxel_value:
                vessel_mask[current_seed] = np.False_
                continue

            current_x, current_y = current_seed[0], current_seed[1]

            adjacent_voxels = []
            possible_moves = [-1, 0, 1]

            for i in possible_moves:
                for j in possible_moves:
                    if i != 0 or j != 0:
                        neighbor = (current_x + i, current_y + j)
                        if not vessel_mask[neighbor]:
                            adjacent_voxels.append(neighbor)

            self.traversed.append(current_seed)

            for neighbor in adjacent_voxels:
                if np.isclose(masked_cross_section[current_seed], masked_cross_section[neighbor], rtol=threshold):
                    vessel_mask[neighbor] = np.True_
                    if neighbor not in stack and self.__check_dim(neighbor):
                        stack.append(neighbor)

    def __segment_arteries(self) -> np.array:

        """
        This method generates the vessel mask using the Frangi vesselness filtering. Since it generates noisy results
        it is protected and will be called from another method which is coded below. This method calls uses
        multiprocessing to increase the calculations speed.

        :return: vessel mask (which by the way can be noisy)
        """

        first_axis, second_axis, third_axis = np.zeros(self.shape), np.zeros(self.shape), np.zeros(self.shape)
        first_axis_length, second_axis_length, third_axis_length = self.shape[0], self.shape[1], self.shape[2]

        def frangi_filter(T1w_cross_section: np.array, INV2_cross_section: np.array, axis: str,
                          layer_index: int) -> None:

            T1w_result = frangi(T1w_cross_section, sigmas=np.arange(.5, 1.5, .25), alpha=0.9, beta=0.9,
                                gamma=40, black_ridges=False)
            INV2_result = frangi(INV2_cross_section, sigmas=np.arange(.5, 1.5, .25), alpha=0.9, beta=0.9,
                                 gamma=40, black_ridges=False)

            try:
                T1w_threshold = np.percentile(T1w_result[np.nonzero(T1w_result)], 99)
            except IndexError:
                T1w_threshold = 0

            try:
                INV2_threshold = np.percentile(INV2_result[np.nonzero(INV2_result)], 97)
            except IndexError:
                INV2_threshold = 0

            T1w_result = T1w_result > T1w_threshold
            INV2_result = INV2_result > INV2_threshold

            layer_dataframe = pd.DataFrame({'T1w': T1w_cross_section.ravel(), 'INV2': INV2_cross_section.ravel()})
            correlation_matrix = layer_dataframe.corr()
            try:
                eigen_values = np.linalg.eig(correlation_matrix)[0]
                T1w_weight, INV2_weight = eigen_values[0], eigen_values[1]
            except np.linalg.LinAlgError:
                T1w_weight, INV2_weight = 1, 1

            T1w_weight, INV2_weight = INV2_weight, T1w_weight

            numerator = (T1w_weight * T1w_cross_section) + (INV2_weight * INV2_cross_section)
            denominator = (T1w_weight + INV2_weight)
            masked_cross_section = numerator / denominator

            try:
                voxel_threshold = np.percentile(masked_cross_section[np.nonzero(masked_cross_section)], 90)
            except IndexError:
                voxel_threshold = 0

            masked_frangi_results = T1w_result | INV2_result

            vessel_mask = (masked_cross_section > voxel_threshold)
            filtered_vessels = masked_frangi_results & vessel_mask

            nonzero_elements = np.nonzero(filtered_vessels)
            vessel_seeds = list(zip(nonzero_elements[0], nonzero_elements[1]))

            for seed in vessel_seeds:
                self.__region_growing(given_seed=seed,
                                      vessel_mask=filtered_vessels,
                                      masked_cross_section=masked_cross_section,
                                      threshold=config_file['PARAMETERs']['region_growing_threshold'],
                                      voxel_value=voxel_threshold)

            if axis == 'first':
                first_axis[layer_index, :, :] = filtered_vessels
            if axis == 'second':
                second_axis[:, layer_index, :] = filtered_vessels
            if axis == 'third':
                third_axis[:, :, layer_index] = filtered_vessels

        parameters_along_first_axis = [(self.T1w[i, :, :], self.INV2[i, :, :], 'first', i) for i in range(first_axis_length)]
        parameters_along_second_axis = [(self.T1w[:, i, :], self.INV2[:, i, :], 'second', i) for i in range(second_axis_length)]
        parameters_along_third_axis = [(self.T1w[:, :, i], self.INV2[:, :, i], 'third', i) for i in range(third_axis_length)]

        time_of_start = time.time()
        console.log('[bold green]applying vesselness filtering...')
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_first_axis)
        console.log('[bold green]step 1 of 3 completed.')
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_second_axis)
        console.log('[bold green]step 2 of 3 completed.')
        Pool(processes=4 * os.cpu_count()).starmap(frangi_filter, parameters_along_third_axis)
        console.log('[bold green]step 3 of 3 completed.')

        console.log('[bold green]total elapsed time {} seconds'.format(round(time.time() - time_of_start)))

        merged_result = ((first_axis.astype(int) + second_axis.astype(int) + third_axis.astype(int)) >= 2)

        return merged_result

    def segment_arteries(self) -> np.array:

        """
        This method calls the private method of above. It requires the user to perform the FieldTrip segmentation on
        both T1w and INV2 modalities in advance, because it uses some of its results to mask out unwanted noises in the
        raw vessel mask obtained by the above method.

        :return: vessel mask
        """

        T1w_soft_tissue_path = os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                            'SOFT_TISSUE_T1w.nii')
        T1w_skull_path = os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                      'SKULL_T1w.nii')
        INV2_soft_tissue_path = os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                             'SOFT_TISSUE_INV2.nii')
        INV2_skull_path = os.path.join(config_file['PATHs']['FieldTrip_RESULTS_PATH'],
                                       'SKULL_INV2.nii')

        try:
            T1w_soft_tissue = nib.load(T1w_soft_tissue_path).get_fdata()
            T1w_skull = nib.load(T1w_skull_path).get_fdata()
            INV2_soft_tissue = nib.load(INV2_soft_tissue_path).get_fdata()
            INV2_skull = nib.load(INV2_skull_path).get_fdata()
        except FileNotFoundError:
            console.log('[bold red]The required files for cleaning the vessel mask are not found.')
            exit()

        segmented_vessels = self.__segment_arteries()
        extra_voxels = T1w_soft_tissue + INV2_soft_tissue + T1w_skull + INV2_skull
        mask = np.zeros_like(extra_voxels, dtype=bool)
        mask[extra_voxels == 0] = True
        dismissal_portion = config_file['PARAMETERs']['dismiss_soft_tissue_portion']
        lower_boundary, upper_boundary = max(dismissal_portion), min(dismissal_portion)
        spared_area_boundaries = (self.shape[2] // lower_boundary, self.shape[2] // upper_boundary)
        mask[:, :, spared_area_boundaries[0]: spared_area_boundaries[1]] = True
        cleaned_vessels = np.where(mask, segmented_vessels, 0)

        if config_file['PARAMETERs']['drop_intensively'] is True:
            cleaned_vessels = SegmentArteries.__filter_connected_components(array=cleaned_vessels,
                                                                            length_threshold=config_file['PARAMETERs']['connected_components_minimum_length'])

        if config_file['PARAMETERs']['save_as_nifti'] is True:
            vessel_nifti = nib.Nifti1Image(cleaned_vessels, self.affine_transform)
            nib.save(vessel_nifti, os.path.join(config_file['PATHs']['RESULTS_PATH'], "vessel_mask.nii"))

        return cleaned_vessels
