import os
import torch
import numpy as np
from osgeo import gdal
import shutil
import torch.nn.functional as F

# Input folder containing paired patches
input_dir = r'/home/hewit_leo/Downloads/aviris_ng_collocated/Very_FInal_by_translate/hr_lr_tiles/'

# Output folder for npz files
output_dir = os.path.join(os.getcwd(), 'data', 'patches_aviris_collocated')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

def read_tiff(path):
    ds = gdal.Open(path)
    arr = ds.ReadAsArray().astype(np.float32)  # C,H,W
    arr[arr == -9999] = 0
    arr[arr < 0] = 0
    arr[arr > 1] = 0
    # Normalize
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = np.transpose(arr, (1,2,0))  # H,W,C
    # if np.all(arr == 0):
    #     return None
    # else:
    #     return arr
    return arr

patch_count = 0

# find all patch1 files
patch1_files = sorted([f for f in os.listdir(input_dir) if f.startswith('patch1_') and f.endswith('.tif')])

for f1 in patch1_files:
    # derive patch2 filename
    f2 = f1.replace('patch1', 'patch2')
    path1 = os.path.join(input_dir, f1)
    path2 = os.path.join(input_dir, f2)
    
    if not os.path.exists(path2):
        print(f"Warning: corresponding {f2} not found, skipping")
        continue
    
    hr_arr = read_tiff(path1)
    lr_arr = read_tiff(path2)

    if np.all(hr_arr == 0) or np.all(lr_arr == 0):
        print(f"Skipping patch {f1} because it's all zeros")
        continue
    
    hr_tensor = torch.from_numpy(np.transpose(hr_arr, (2,0,1))).float()  # C,H,W
    lr_tensor = torch.from_numpy(np.transpose(lr_arr, (2,0,1))).float()  # C,H,W

    # save as npz
    np.savez(os.path.join(output_dir, f'aviris_collocated_hr_lr_patch_{patch_count}.npz'),
             hr=hr_tensor, lr=lr_tensor)
    patch_count += 1

print(f"Total paired patches saved: {patch_count}")
