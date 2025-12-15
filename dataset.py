import os, scipy, scipy.io, h5py, torch, shutil, numpy as np
from osgeo import gdal
import torch.nn.functional as F

# directories
directory = os.path.join(os.getcwd(), 'data', 'original_data') #origial data
train_val_test_save_dir = os.path.join(os.getcwd(), 'data','patches') # train, validation and test patches without split

# to delete the existing files before running the code
for path in [train_val_test_save_dir]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def normalizer(array):
    norm_arr = ((array - array.min()) / (array.max() - array.min() + 1e-8))
    return norm_arr

def downsampler(tensor, downsample_scale): 
    tensor = tensor.unsqueeze(0)
    ds = F.interpolate(tensor, tensor.shape[2] // downsample_scale, mode = 'bicubic')
    ds = ds.squeeze(0)
    return ds

# to read mat/tiff and create the patches for chikusei scene, pavia centre scene, washinton dc dataset
def read_mat(dir, dataset_name):

    # for pavia centre scene
    if dataset_name == 'pavia_centre':
        image = os.path.join(dir, 'pavia_centre.mat')
        image_array = scipy.io.loadmat(image)
        image_array = image_array['pavia']
        image_array = normalizer(image_array)
        print('Input shape is : ', image_array.shape)
        return image_array


    # for chikusei data
    elif dataset_name == 'chikusei':
        filepath = os.path.join(dir, 'chikusei.mat')
        with h5py.File(filepath, 'r') as file:
            # image_array = normalizer(np.array(file['chikusei']))
            image_array = np.array(file['chikusei'], dtype=np.float32)
            image_array = image_array[:,144:2191,107:2410]
            image_array = normalizer(image_array)
            image_array_test = image_array[:, :128, :]
            image_array_train = image_array[:, 128:, :]
            image_array_train = np.transpose(image_array,(1,2,0))
            image_array_test = np.transpose(image_array_test, (1,2,0))
            print('Min value:',image_array_train.min(), 'Max Value:',image_array_train.max())
            print(f'Train shape: {image_array_train.shape}, Test shape: {image_array_test.shape}')
        return image_array_train, image_array_test


    # for washinton dc dataset
    elif dataset_name == 'washinton_dc':
        path = os.path.join(dir, 'washinton_dc.tif')
        image_opn = gdal.Open(path)
        image_array = image_opn.ReadAsArray()
        image_array = image_array.astype(np.float32)
        image_array = normalizer(image_array)
        print('Input shape is : ', image_array.shape)
        return image_array


    # for aviris ng
    elif dataset_name == 'aviris_ng':
        path = os.path.join(dir, 'aviris_ng.tif') ##c, h, w
        image_opn = gdal.Open(path)
        image_array = image_opn.ReadAsArray()
        print("Raw shape of aviris ng:", image_array.shape)
        image_array = image_array.astype(np.float32)
        image_array[image_array == -9999] = 0
        image_array[image_array < 0 ] = 0
        image_array[image_array > 1] = 0
        image_array = normalizer(image_array)
        image_array = np.transpose(image_array,(1,2,0))
        print('Input shape is : ', image_array.shape)
        return image_array


def patch_extractor(image_arr, patch_size, stride, dataset_name):
    if dataset_name == 'chikusei':
        n = 0
        for i in range (0, image_arr.shape[0] - patch_size + 1, stride):
            for j in range (0, image_arr.shape[1] - patch_size + 1, stride):
                print("Inside patch_extractor:", image_arr.shape, image_arr.ndim)
                move = image_arr[i : i + patch_size, j : j + patch_size, :]
                if move.shape[0] == move.shape[1]:
                    move = np.transpose(move,(2,0,1))
                    move = torch.from_numpy(move).float()
                    hr_patch = move #normalizer(move)
                    # print(hr_patch.min(), hr_patch.max())
                    lr_patch = downsampler(hr_patch, downsample_scale=downsample_size)
                    lr_patch = torch.clamp(lr_patch, 0, 1)
                    # print(lr_patch.min(), lr_patch.max())
                    np.savez(os.path.join(train_val_test_save_dir, f'hr_lr_patch_{n}.npz'), lr = lr_patch, hr = hr_patch)
                    n += 1


    elif dataset_name == 'pavia_centre':
        n = 0
        for i in range (0, image_arr.shape[0] - patch_size + 1, stride):
            for j in range (0, image_arr.shape[1] - patch_size + 1, stride):
                move = image_arr[i : i+patch_size, j : j+patch_size, :]
                if move.shape[0] == move.shape[1]:
                    move = np.transpose(move,(2,0,1))
                    move = torch.from_numpy(move).float()
                    hr_patch = move
                    lr_patch = downsampler(hr_patch, downsample_scale=downsample_size)
                    lr_patch = torch.clamp(lr_patch, 0, 1)
                    np.savez(os.path.join(train_val_test_save_dir, f'{dataset_name}_hr_lr_patch_{n}.npz'), lr = lr_patch, hr = hr_patch)
                    n += 1

    elif dataset_name == 'washinton_dc':
        n = 0
        for i in range (0, image_arr.shape[1] - patch_size + 1, stride):
            for j in range (0, image_arr.shape[2] - patch_size + 1, stride):
                move = image_arr[:, i : i+patch_size, j : j+patch_size]
                if move.shape[1] == move.shape[2]:
                    # move = np.transpose(move,(2,0,1))
                    move = torch.from_numpy(move).float()
                    hr_patch = move
                    lr_patch = downsampler(hr_patch, downsample_scale=downsample_size)
                    lr_patch = torch.clamp(lr_patch, 0, 1)
                    np.savez(os.path.join(train_val_test_save_dir, f'{dataset_name}_hr_lr_patch_{n}.npz'), lr = lr_patch, hr = hr_patch)
                    n += 1

    elif dataset_name == 'washinton_dc':
        n = 0
        for i in range (0, image_arr.shape[1] - patch_size + 1, stride):
            for j in range (0, image_arr.shape[2] - patch_size + 1, stride):
                move = image_arr[:, i : i+patch_size, j : j+patch_size]
                if move.shape[1] == move.shape[2]:
                    move = np.transpose(move,(2,0,1))
                    move = torch.from_numpy(move).float()
                    hr_patch = move
                    lr_patch = downsampler(hr_patch, downsample_scale=downsample_size)
                    lr_patch = torch.clamp(lr_patch, 0, 1)
                    np.savez(os.path.join(train_val_test_save_dir, f'{dataset_name}_hr_lr_patch_{n}.npz'), lr = lr_patch, hr = hr_patch)
                    n += 1

    elif dataset_name == 'aviris_ng':
        n = 0
        for i in range (0, image_arr.shape[0] - patch_size + 1, stride):
            for j in range (0, image_arr.shape[1] - patch_size + 1, stride):

                print("Inside patch_extractor:", image_arr.shape, image_arr.ndim)

                move = image_arr[i : i+patch_size, j : j+patch_size, :]
                if move.shape[0] == move.shape[1]:
                    move = np.transpose(move,(2,0,1))
                    move = torch.from_numpy(move).float()
                    hr_patch = move
                    lr_patch = downsampler(hr_patch, downsample_scale=downsample_size)
                    lr_patch = torch.clamp(lr_patch, 0, 1)
                    if not torch.all(hr_patch == 0):
                        print(hr_patch.shape)   
                        print(lr_patch.shape)
                        np.savez(os.path.join(train_val_test_save_dir, f'{dataset_name}_hr_lr_patch_{n}.npz'), lr = lr_patch, hr = hr_patch)
                    n += 1

#=====================================================================================
# here mention the dataset name, patch size and stride

dataset_name = 'chikusei' #type any one name 'chikusei' or 'washinton_dc' or 'pavia_centre' or 'aviris_ng'
patch_size = 64
stride = 32
downsample_size = 2

patch_extractor(read_mat(directory, dataset_name=dataset_name)[0], patch_size=patch_size, stride=stride, dataset_name=dataset_name) #it also save the hr and lr patches
# patch_extractor(read_mat(directory, dataset_name=dataset_name)[1], patch_size=128, stride=128, dataset_name=dataset_name) #test alone big patches

print(f'Total number of train and val patches are {len(os.listdir(train_val_test_save_dir))}')
