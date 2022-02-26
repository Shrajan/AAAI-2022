'''
This script converts the Prostate Segmentation MRI data and ground-truth labels to the required format.
'''

from os import mkdir
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import argparse

'''
Functions "get3dslice" and parts of "resample_4D_images" are from https://discourse.itk.org/t/resampleimagefilter-4d-images/2172/2
Author: SachidanandAlle
Date Taken: 2 July 2021
'''
def get3dslice(image, slice=0):
    size = list(image.GetSize())
    if len(size) == 4:
        size[3] = 0
        index = [0, 0, 0, slice]

        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size)
        extractor.SetIndex(index)
        image = extractor.Execute(image)
    return image

def resample_4D_images(input_image, new_spacing, interpolation="trilinear"):
    """
    input image will be resampled.
    """    
    # Resample 4D (SITK Doesn't support directly; so iterate through slice and get it done)
    #new_data_list = []
    size = list(input_image.GetSize())
    for s in range(size[3]):
        img = get3dslice(input_image, s)
        img = resample_3D_images(input_image=img, new_spacing=new_spacing, interpolation=interpolation)
        #new_data_list.append(img)
        break # Get only the first slice T2 modality. 

    #joinImages = sitk.JoinSeriesImageFilter()
    #newimage = joinImages.Execute(new_data_list)
    newimage = img
    return newimage

def resample_3D_images(input_image, new_spacing, interpolation="trilinear"):
    """
    input image will be resampled.
    """    
    # Resample image.
    resample = sitk.ResampleImageFilter()
    
    if interpolation == "trilinear":
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolation == "nearest":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    orig_size = np.array(input_image.GetSize(), dtype=np.int32)
    orig_spacing = input_image.GetSpacing()
    new_size = orig_size*([i / j for i, j in zip(list(orig_spacing),list(new_spacing))])
    new_size = np.ceil(new_size).astype(np.int32)
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(input_image)

    return newimage


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, default="Task05_Prostate", required=False, help="Folder/directory to read the original MSD prostate data.")
    parser.add_argument("--out_folder", type=str, default="mri_framework/train_and_test", required=False, help="Folder/directory to save the converted data.")
    parser.add_argument("--change_spacing", action='store_true', help="If set, then data and corresponding label will be resampled to new_spacing.")
    parser.add_argument("--new_spacing", type=float, nargs=3, default=(0.6, 0.6, 0.6), required=False, help="Spacing to be resampled.")
    opt = parser.parse_args()
    
    in_folder = opt.in_folder
    out_folder = opt.out_folder

    # Create output folder.
    if not os.path.exists(os.path.join(out_folder)):
        os.makedirs(os.path.join(out_folder))

    # Get the paths of the data and labels.
    raw_data = subfiles(join(in_folder,"imagesTr"), suffix=".nii.gz")
    segmentations = subfiles(join(in_folder,"labelsTr"), suffix=".nii.gz")

    # Resample, change format to nrrd and save data and label in individually folders.
    if opt.change_spacing == True:
        for index, (dataPath, labelPath) in enumerate(zip(raw_data, segmentations)):
            mkdir(join(out_folder,str(index)))
            print("\nFolder number: " + str(index) + " " + dataPath + " " + labelPath)
            
            old_data = sitk.ReadImage(dataPath)
            new_data = resample_4D_images(input_image=old_data, new_spacing=opt.new_spacing)
            data_fname = join(out_folder,str(index), "data.nrrd")
            sitk.WriteImage(new_data, data_fname)
            
            old_label = sitk.ReadImage(labelPath)
            new_label = resample_3D_images(input_image=old_label, new_spacing=opt.new_spacing, interpolation = "nearest")
            label_fname = join(out_folder,str(index), "label.nrrd")
            sitk.WriteImage(new_label, label_fname)

            print("The old data image has shape: " + str(old_data.GetSize()) + " with spacing: " + str(old_data.GetSpacing()))
            print("The new data image has shape: " + str(new_data.GetSize()) + " with spacing: " + str(new_data.GetSpacing()))
            print("The old label image has shape: " + str(old_label.GetSize()) + " with spacing: " + str(old_label.GetSpacing()))
            print("The new label image has shape: " + str(new_label.GetSize()) + " with spacing: " + str(new_label.GetSpacing()))

    #  No resampling, change format to nrrd and save data and label in individually folders.
    else:
        for index, (dataPath, labelPath) in enumerate(zip(raw_data, segmentations)):
            mkdir(join(out_folder,str(index)))
            print("Folder number: " + str(index) + " " + dataPath + " " + labelPath)
            
            data_fname = join(out_folder,str(index), "data.nrrd")
            sitk.WriteImage(sitk.ReadImage(dataPath), data_fname)
            
            label_fname = join(out_folder,str(index), "label.nrrd")
            sitk.WriteImage(sitk.ReadImage(labelPath), label_fname)
