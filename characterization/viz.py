import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def plot_slice(image_path):
    img = nib.load(image_path).get_fdata()
    print(f"Image shape: {img.shape}")
    plt.imshow(img[:, :, img.shape[2] // 2], cmap='gray')


def parse_voxel_spacing(vs):
    if isinstance(vs, str):
        return tuple(map(float, vs.strip("()").split(",")))
    return vs

def compute_anisotropy_ratio(vs):
    vs = np.array(vs)
    return np.max(vs) / np.min(vs)

def categorize_anisotropy(ratio, iso_threshold=1.0, high_aniso_threshold=2.0):
    if ratio == iso_threshold:
        return 'Isotropic'
    elif ratio < high_aniso_threshold:
        return 'Mildly Anisotropic'
    else:
        return 'Highly Anisotropic'