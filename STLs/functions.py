from skimage import measure
import nibabel as nib
from stl import mesh
import numpy as np


def create_stl_file_from_array(given_array, output_file_name):

    """
    This function creates a 3D object from a given array and saves it as an STL file.

    :param given_array: The array from which the 3D object will be created.
    :param output_file_name: The name of the output file.
    :return: None
    """

    vertices, faces, normals, values = measure.marching_cubes(given_array, 0)

    object_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        object_3d.vectors[i] = vertices[f]

    object_3d.save(f'{output_file_name}.stl')


def create_stl_file_from_nifti(file_path, output_file_name):

    """
    This function creates a 3D object from a given NIFTI file and saves it as an STL file.
    
    :param file_path: The path of the NIFTI file.
    :param output_file_name: The name of the output file.
    :return: None
    """

    nifti_file_array = nib.load(file_path).get_fdata()

    vertices, faces, normals, values = measure.marching_cubes(nifti_file_array, 0)

    object_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        object_3d.vectors[i] = vertices[f]

    object_3d.save(f'stl\\{output_file_name}.stl')
