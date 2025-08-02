import os
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.ndimage
import shutil
import warnings
import re
import ants
import SimpleITK as sitk
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance, ttest_ind
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from sklearn.cluster import MiniBatchKMeans # Not directly used in this version

# Required for tf.keras.models.Model
from tensorflow.keras.models import Model
# --- PREPROCESSING FUNCTION (UPDATED WITH CACHING AND T1 FILTER) ---
def preprocess_nifti_file(input_path: str, output_dir: str, template_path: str, modality: str, force_overwrite: bool = False):
    """
    Executes a complete preprocessing pipeline for a single NIfTI file.
    - Adds a check to skip processing if the output file already exists and force_overwrite is False.
    - Adds an explicit check for 'T1w' modality.
    """
    output_filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, output_filename)

    # Check 1: Skip if already preprocessed and not forcing overwrite
    if os.path.exists(output_path) and not force_overwrite:
        print(f"--- Skipping {output_filename}: Already preprocessed. ---")
        return output_path

    # Check 2: Process only T1-weighted images
    if modality.lower() != 't1w': # Assuming 'T1w' is the standard for T1-weighted. Adjust if needed.
        print(f"--- Skipping {output_filename}: Modality is '{modality}', but only 'T1w' is supported. ---")
        return None

    print(f"--- Starting preprocessing for file: {output_filename} (Modality: {modality}) ---")
    
    try:
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"MNI template not found at {template_path}")

        # 1) Loading image
        print("    [1/5] Loading image...")
        img = ants.image_read(input_path, reorient='IAL') # Reorient to IAL for consistent spatial interpretation
        
        # Initial brain mask for bias correction (can be refined later for skull stripping)
        # It's good to have an initial mask for N4, as it works better within brain tissue.
        initial_brain_mask = ants.get_mask(img) 

        # 2) Bias Field Correction (N4) - Applied before registration for better intensity consistency
        print("    [2/5] Bias Field Correction (N4)...")
        # Convert ANTs image to SimpleITK for N4
        img_array_for_sitk = img.numpy()
        sitk_img = sitk.GetImageFromArray(img_array_for_sitk)
        
        # Transfer spatial information from ANTs to SimpleITK
        sitk_img.SetSpacing(img.spacing)
        sitk_img.SetOrigin(img.origin)
        sitk_img.SetDirection(img.direction[:3, :3].flatten()) # Use 3x3 rotation part

        mask_array_for_sitk = initial_brain_mask.numpy().astype(np.uint8)
        sitk_mask = sitk.GetImageFromArray(mask_array_for_sitk)
        sitk_mask.SetSpacing(initial_brain_mask.spacing)
        sitk_mask.SetOrigin(initial_brain_mask.origin)
        sitk_mask.SetDirection(initial_brain_mask.direction[:3, :3].flatten())

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_sitk = corrector.Execute(sitk_img, sitk_mask)

        corrected_numpy_array = sitk.GetArrayFromImage(corrected_sitk)
        
        # Convert back to ANTs image, preserving original spatial info from 'img'
        n4_corrected_img = ants.from_numpy(
            corrected_numpy_array, 
            origin=img.origin,  # Use original spatial info
            spacing=img.spacing, 
            direction=img.direction
        )
        # Apply initial mask again, as N4 can sometimes introduce intensity outside the mask
        n4_corrected_img = n4_corrected_img * initial_brain_mask

        # 3) Registration to MNI template
        print("    [3/5] Registration to MNI template...")
        template_img = ants.image_read(template_path, reorient='IAL')
        
        # Perform registration using the bias-corrected image
        transform = ants.registration(
            fixed=template_img,
            moving=n4_corrected_img,
            type_of_transform='SyN', # SyN is a powerful deformable registration
        )
        registered_img = ants.apply_transforms(
            fixed=template_img,
            moving=n4_corrected_img,
            transformlist=transform['fwdtransforms'],
            interpolator='linear'
        )
        
        # 4) Brain Extraction (Skull Stripping) - Apply mask of the *registered* image
        print("    [4/5] Brain Extraction (Skull Stripping)...")
        # Use a more robust brain extraction method if possible, or apply a brain mask derived from template.
        # For simplicity here, we'll use a mask from the *registered* image, or a template mask.
        # Using the template's brain mask after registration is generally a robust approach.
        template_brain_mask = ants.get_mask(template_img) # Get mask from the MNI template
        skull_stripped_img = registered_img * template_brain_mask # Apply template mask to registered image

        # 5) Intensity Normalization
        print("    [5/5] Intensity Normalization...")
        # Normalization should be applied to the skull-stripped image
        img_array = skull_stripped_img.numpy()
        
        # Only consider pixels within the brain mask for normalization
        valid_pixels = img_array[img_array > 0] # Assuming 0 is background after skull stripping

        min_val = np.min(valid_pixels) if valid_pixels.size > 0 else 0
        max_val = np.max(valid_pixels) if valid_pixels.size > 0 else 1 # Avoid division by zero if all pixels are same

        if max_val > min_val:
            normalized_img_array = (img_array - min_val) / (max_val - min_val)
        else:
            normalized_img_array = np.zeros_like(img_array) # If all pixels are same or no valid pixels

        # Reconstruct ANTs image with normalized array, preserving spatial info from skull_stripped_img
        normalized_img = ants.from_numpy(
            normalized_img_array, 
            origin=skull_stripped_img.origin,
            spacing=skull_stripped_img.spacing, 
            direction=skull_stripped_img.direction
        )
        
        # Re-apply mask to ensure background is zero after normalization
        normalized_img = normalized_img * template_brain_mask 

        print(f"    -> Saving result to: {output_path}")
        ants.image_write(normalized_img, output_path)
        
        print(f"--- Preprocessing for {output_filename} completed successfully. ---\n")
        return output_path

    except Exception as e:
        print(f"!!!!!! ERROR processing file {output_filename}: {e} !!!!!!\n")
        return None