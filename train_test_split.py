import os 
import shutil
from sklearn.model_selection import train_test_split

#patches hr and lr
lr_hr_patches_path = os.path.join(os.path.dirname(__file__), 'data','patches')

# lr_hr_patches_path = r'/home/hewit_leo/PG/Project/my_architecture_1/data/patches_aviris_collocated' ##4.1 and 8.2

#destination
train_path = os.path.join(os.path.dirname(__file__), 'dataset','train')
validation_path = os.path.join(os.path.dirname(__file__), 'dataset','validation')
test_path = os.path.join(os.path.dirname(__file__), 'dataset','test')

for path in [train_path, validation_path, test_path]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

#file list
lr_hr_patches_files = sorted(os.listdir(lr_hr_patches_path))
print(lr_hr_patches_files)
print("Process is going on, wait")
total_patches_to_use = lr_hr_patches_files#[:800] #------------- mention how many files want

def train_test_spliter(patches):
    train, _ = train_test_split(patches, test_size=0.2, random_state=42, shuffle=True)
    val, test = train_test_split(_, test_size=0.1, random_state=42, shuffle=True)
    for i in train:
        shutil.copyfile(os.path.join(lr_hr_patches_path, i), os.path.join(train_path, i))
    for j in val:
        shutil.copyfile(os.path.join(lr_hr_patches_path, j), os.path.join(validation_path, j))
    for k in test:
        shutil.copyfile(os.path.join(lr_hr_patches_path, k), os.path.join(test_path, k))

train_test_spliter(total_patches_to_use)
print(f'Total number of train patches are {len(os.listdir(train_path))}')
print(f'Total number of val patches are {len(os.listdir(validation_path))}')
print(f'Total number of test patches are {len(os.listdir(test_path))}')