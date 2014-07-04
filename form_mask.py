# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:27:05 2014

@author: sb238920
"""
# Standard library imports
import os

# Related third party imports
import numpy as np
import nibabel as nib

# Local application/library specific imports


def create_binary_union(mask_paths, union_path):
    """ Creates the binary mask obtained by union of given masks of same
    shape and returns the path to the created mask.
    """
    for n, mask_path in enumerate(mask_paths):
        img = nib.load(mask_path)
        the_data = img.get_data()
        if n == 0:
            shape = the_data.shape
            union_data = np.zeros(shape, dtype=np.int16)
        if the_data.shape != shape:
            raise ValueError("mask {0} has not the correct shape".format(n))

        union_data += the_data
        union_data[union_data > 0] = 1
    img = nib.Nifti1Image(union_data, np.eye(4))
    img.to_filename(union_path)


def create_labels_img(mask_paths, labels_img_path):
    """ Creates the label image obtained from different masks of same
    shape and returns the path to the created image.
    """
    n_labels = len(mask_paths)
    for n, mask_path in enumerate(mask_paths):
        img = nib.load(mask_path)
        the_data = img.get_data()
        if n == 0:
            shape = the_data.shape
            affine = img.get_affine()
            labels_data = np.zeros(shape + (n_labels,), dtype=np.int16)
        if the_data.shape != shape:
            raise ValueError("mask {0} has not the correct shape".format(n))
        labels_data[..., n] += the_data
#    labels_data[labels_data > 0] = 1
    labels_data = np.array(labels_data)
    img = nib.Nifti1Image(labels_data, affine)
    img.to_filename(labels_img_path)


if __name__ == "__main__":
    data_path = "/volatile/new/salma/conn/genereated_rois"
    labels_img = os.path.join(data_path, "Visual_AN_DMN_labels_ordered.nii")
    if not os.path.isfile(labels_img):
        mask_paths = [
                      "ventralPrimaryVisualCortexLR_sphere6mm.nii",
                      "dorsalPrimaryVisualCortexLR_sphere6mm.nii",
                      "ventralIntraparietalSulcusLR_sphere6mm.nii",
                      "posteriorIntraparietalSulcusLR_sphere6mm.nii",
                      "middleTemporalRegionLR_sphere6mm.nii",
                      "frontalEyeFieldLR_sphere6mm.nii",
                      "temporoparietalJunctionR_sphere6mm.nii",
                      "dorsolateralPrefrontalCortexR_sphere6mm.nii",
                      "angularGyrusLR_2clusters_sphere6mm.nii",
                      "superiorFrontalGyrusLR_2clusters_sphere6mm.nii",
                      "posteriorCingulate_sphere6mm.nii",
                      "MPFC_sphere6mm.nii",
                      "frontopolarCortex_sphere6mm.nii"]
        mask_paths = [os.path.join(data_path, mask) for mask in mask_paths]
        create_labels_img(mask_paths, labels_img)

    union_path = os.path.join(data_path, "WMN_AN_DMN.nii")
    if not os.path.isfile(union_path):
        mask_paths = ["WMN_final/inferiorParietalLobule_sphere6mm.nii",
                      "WMN_final/middleFrontalGyrusL_peak1_sphere6mm.nii",
                      "WMN_final/cerebellumPosteriorLobeR_peak1_sphere6mm.nii",
                      "WMN_final/cerebellumPosteriorLobeL_peak3_sphere6mm.nii",
                      "WMN_final/thalamusL_sphere6mm.nii",
                      "angularGyrusLR_2clusters_sphere6mm.nii",
                      "superiorFrontalGyrusLR_2clusters_sphere6mm.nii",
                      "posteriorCingulate_sphere6mm.nii",
                      "MPFC_sphere6mm.nii",
                      "frontopolarCortex_sphere6mm.nii",
                      "ventralIntraparietalSulcusLR_sphere6mm.nii",
                      "temporoparietalJunctionR_sphere6mm.nii",
                      "dorsolateralPrefrontalCortexR_sphere6mm.nii",
                      "posteriorIntraparietalSulcusLR_sphere6mm.nii",
                      "middleTemporalRegionLR_sphere6mm.nii",
                      "frontalEyeFieldLR_sphere6mm.nii"]
        mask_paths = [os.path.join(data_path, mask) for mask in mask_paths]
        create_binary_union(mask_paths, union_path)