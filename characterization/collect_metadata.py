import os
import nibabel as nib
import numpy as np
import pandas as pd


def check_schema(sample: dict) -> bool:
    if not isinstance(sample, dict):
        raise ValueError("data_dict must be a dictionary")
    
    expected_keys = {
        "image_id"     : str,
        "dataset"      : str,
        "patient"      : str,
        "image_path"   : str,
        "modality"     : str,
        "shape"        : tuple,
        "voxel_spacing": tuple,
        "axcodes"      : (tuple, list, str),
        "min_value"    : float,
        "max_value"    : float,
        "median_value" : float,
        "affine"       : list,
        "orientation"  : list,
        "is_hwd"       : bool,
        "preprocessed_image_path": str,
    }

    for key, expected_type in expected_keys.items():
        if key not in sample:
            print(f"Missing key: {key}")
            return False
        if not isinstance(sample[key], expected_type):
            print(f"Key '{key}' has wrong type: expected {expected_type}, got {type(sample[key])}")
            return False
    return True

def get_paths(directory, file_extension=".nii.gz", file_extension_list=None):
    paths = []
    for root, _, files in os.walk(directory):
        for file in files:

            if file_extension_list:
                for file_extension in file_extension_list:
                    if file.endswith(file_extension):
                        paths.append(os.path.join(root, file))
            else:
                if file.endswith(file_extension):
                    paths.append(os.path.join(root, file))
                    
    return paths

def append_dict_to_csv(data_dict, csv_file):
    # Check if CSV file exists, if not create it
    try: 
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(data=[data_dict], columns=data_dict.keys())
        df.to_csv(csv_file, index=False)
        return

    # If CSV file exists, append the new data if the image_id is not there (unique image_id)
    existing_ids = set(df['image_id'].tolist())
    new_id = data_dict['image_id']

    if new_id not in existing_ids:
        new_row_df = pd.DataFrame(data=[data_dict])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(csv_file, index=False)

def extract_image_attributes(image_path):
    # Load image
    try:
        img  = nib.load(image_path)
    except Exception:
        print(f"Failed loading image {image_path}. Cannot extract image attributes.")
        return {}
    data = img.get_fdata()

    # Extract image attributes
    affine        = img.affine
    ornt          = nib.orientations.io_orientation(affine)
    axcodes       = "".join(nib.orientations.aff2axcodes(affine))
    voxel_spacing = tuple([float(val) for val in img.header.get_zooms()])
    is_hwd        = (axcodes == 'RAS') and (np.array_equal(ornt[:, 0], [0, 1, 2]))

    return {
        "shape"        : data.shape,
        "voxel_spacing": voxel_spacing,
        "axcodes"      : axcodes,
        "min_value"    : float(np.min(data)),
        "max_value"    : float(np.max(data)),
        "median_value" : float(np.median(data)),
        "affine"       : affine.tolist(),
        "orientation"  : ornt.tolist(),
        "is_hwd"       : is_hwd
    }

def collect_metadata(image_paths, csv_file, extract_nonimage_attributes_fn):
    n = len(image_paths)
    for i, image_path in enumerate(image_paths):
        image_attributes = extract_image_attributes(image_path)
        non_image_attributes = extract_nonimage_attributes_fn(image_path)

        metadata = {
            **image_attributes,
            **non_image_attributes,
        }

        # Check schema
        if not check_schema(metadata):
            raise ValueError(f"Metadata dictionary for {image_path} does not conform to the expected schema.")
        
        # Append the data dictionary to the CSV file
        append_dict_to_csv(metadata, csv_file)

        print(f"Collected metadata for {i+1}/{n} images.")

if __name__ == "__main__":
    dummy_data = np.random.rand(240, 240, 155)
    dummy_affine = np.eye(4)
    dummy_ornt = np.array([[0, 1], [1, 1], [2, 1]])

    test_sample = {
        "image_id": "img001",
        "dataset": "ExampleDataset",
        "patient": "patient_123",
        "image_path": "/path/to/image.nii.gz",
        "modality": "T1",
        "shape": dummy_data.shape,
        "voxel_spacing": (1.0, 1.0, 1.0),
        "axcodes": ("R", "A", "S"),
        "min_value": float(np.min(dummy_data)),
        "max_value": float(np.max(dummy_data)),
        "median_value": float(np.median(dummy_data)),
        "affine": dummy_affine.tolist(),
        "orientation": dummy_ornt.tolist(),
        "is_hwd": True,
        "preprocessed_image_path": "/path/to/preprocessed/image.npy",

    }

    is_valid = check_schema(test_sample)
    assert is_valid, "Schema validation failed for the test sample."

    del test_sample["image_id"]
    is_valid = check_schema(test_sample)
    assert not is_valid, "Schema validation should fail for the test sample with missing 'image_id'."