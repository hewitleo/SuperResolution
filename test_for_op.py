import torch
import os
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from loss import HybridLoss, mse_sam_combo, OnlySAMLoss, SAM_L1_Loss
from loss import HybridLoss
import torch.utils.data
import time; start = time.time()
from metrics import quality_assessment
from SpatialSR import SpatialSR
from HSISR_new import HSISR_new

seed = 42 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Random seed fixed:", seed)

#=====================================================================================
data_dir = os.path.join(os.path.dirname(__file__),'dataset')
train_path = os.path.join(data_dir, 'train')
validation_path = os.path.join(data_dir, 'validation')
test_path = os.path.join(data_dir, 'test')

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, directory, augment=False):
        self.directory = directory
        self.files = [x for x in os.listdir(directory) if x.endswith('.npz')]
        self.augment = augment

        if self.augment: 
            self.factor = 7
        else:
            self.factor = 1

        original_samples = len(self.files)
        total_samples = original_samples * self.factor
        print(f"Original samples: {original_samples}")
        print(f"Total samples after augmentation: {total_samples}")

    def __len__(self):
        return len(self.files) * self.factor

    def data_augmentation(self, label, mode=0): #from sspsr github
        if mode == 0:
            return label
        elif mode == 1:
            return np.flip(label, axis=1)  # flip vertically
        elif mode == 2:
            return np.rot90(label, k=1, axes=(1, 2))  # rotate 90
        elif mode == 3:
            return np.flip(np.rot90(label, k=1, axes=(1, 2)), axis=1)
        elif mode == 4:
            return np.rot90(label, k=2, axes=(1, 2))  # rotate 180
        elif mode == 5:
            return np.flip(np.rot90(label, k=2, axes=(1, 2)), axis=1)
        elif mode == 6:
            return np.rot90(label, k=3, axes=(1, 2))  # rotate 270
        elif mode == 7:
            return np.flip(np.rot90(label, k=3, axes=(1, 2)), axis=1)

    def __getitem__(self, idx):
        file_index = idx
        aug_num = 0
        if self.augment:
            file_index = idx // self.factor
            aug_num = int(idx % self.factor)

        data = np.load(os.path.join(self.directory, self.files[file_index]))
        hr_patch = data['hr'].astype(np.float32)
        lr_patch = data['lr'].astype(np.float32)

        if self.augment:
            lr_patch = self.data_augmentation(lr_patch, mode=aug_num)
            hr_patch = self.data_augmentation(hr_patch, mode=aug_num)

            # to handle neg values
            lr_patch = lr_patch.copy()
            hr_patch = hr_patch.copy()

        return torch.from_numpy(lr_patch)[:,:,:], torch.from_numpy(hr_patch)[:,:,:]

train_dataset = MyDataSet(train_path, augment=True)
validation_dataset = MyDataSet(validation_path)
test_dataset = MyDataSet(test_path)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
validation_loader = DataLoader(validation_dataset, batch_size = 16, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

# #========================================================================================
# # def check_point(epoch, model, optimizer, scheduler, train_loss, test_loss, device):
# def check_point(epoch, model, optimizer, train_loss, test_loss, device):
#     model.eval().cpu()
#     checkpoint_model_dir = os.path.join(os.path.dirname(__file__), 'model', 'checkpoints')
#     ckpt_model_filename = 'chickusei' + "_" + 'my_net' + "_ckpt_epoch_" + str(epoch) + ".pth"
#     os.makedirs(checkpoint_model_dir, exist_ok=True)
#     ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
#     state = {"epoch": epoch, 
#              "model": model,
#             "model_state_dict": model.state_dict(), 
#             "optimizer_state_dict": optimizer.state_dict(), 
#             # "scheduler_state_dict": scheduler.state_dict(),
#             "train_loss": train_loss,
#             "test_loss": test_loss
#         }
#     torch.save(state, ckpt_model_path)
#     model.to(device).train()
#     print(f"Checkpoint saved to {ckpt_model_path}")
# # #========================================================================================

# def spectral_angle_mapper(gt, pred):
#     gt = gt / np.linalg.norm(gt)
#     pred = pred / np.linalg.norm(pred)
#     return np.arccos(np.clip(np.dot(gt, pred), -1.0, 1.0))

epoch = 40
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpatialSR().to(device)
# model = HSISR_new().to(device)

# print(' ')
# print('Training Started')
# print('===============================================>')
# print('be patient it will take time')
# print(' ')
# print('belive you can improve your model gradually')
# print(' ')
# print('you will definitely make powerful efficient model')
# print(' ')
# print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
# print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
# print(' ')
# # print(model)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#for warm up and cosine annealing
# from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
# warmup_epochs = 5
# cosine_epochs = epoch - warmup_epochs
# warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
# scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch) # #schedular for cosine a lr alone

# ## for step lr
# from torch.optim.lr_scheduler import StepLR
# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# #loss functions
# loss_function = nn.MSELoss()
# l1_loss_fn = nn.L1Loss()
# # h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
# mse_and_sam_loss = mse_sam_combo(N=16)
# loss_sam = OnlySAMLoss()
# loss_sam_l1 = SAM_L1_Loss()

# printed_shapes = False
# for i in range(epoch):
#     model.train()
#     running_loss = 0.0
#     # adjust_learning_rate(start_lr=learning_rate, optimizer=optimizer, epoch=i)
#     lr_now = optimizer.param_groups[0]['lr']
#     for lr_patches, hr_patches in train_loader:
#         if not printed_shapes:
#             print('Lr dim:', lr_patches[0].shape)
#             print('Hr dim:', hr_patches[0].shape)
#             printed_shapes = True

#         lr_patches = lr_patches.to(device)
#         hr_patches = hr_patches.to(device)
#         optimizer.zero_grad()

#         output = model(lr_patches)
#         # loss = mse_and_sam_loss(output, label=hr_patches) #change line 134
#         # loss = loss_function(output, hr_patches)
#         # loss = compare_sam(output, hr_patches)
#         # loss = l1_loss_fn(output, hr_patches)
#         #loss = h_loss(output, hr_patches)
#         # loss = loss_sam(output, hr_patches)
#         loss = loss_sam_l1(output, hr_patches)
#         loss.backward()

#         # for name, param in model.named_parameters():
#         #     if param.grad is not None:
#         #         print(f"{name} - grad norm: {param.grad.norm().item()}")

#         optimizer.step()

#         running_loss += loss.item()
#     avg_loss = running_loss / len(train_loader)

#     print(f'Epoch [{i+1} / {epoch}], Training Loss: {avg_loss:.5f}, Lr: {lr_now}')
    
#     scheduler.step()

#     model.eval()
#     test_loss = 0.0
#     all_metrics = {'PSNR': [], 'SAM': [], 'SSIM': [], 'ERGAS': [], 'CrossCorrelation': []}

#     with torch.no_grad():
#         for lr_patches, hr_patches in validation_loader:
#             lr_patches = lr_patches.to(device)
#             hr_patches = hr_patches.to(device)
#             output = model(lr_patches)
#             # loss = loss_function(output, hr_patches)
#             #loss = h_loss(output, hr_patches)
#             loss = loss_sam_l1(output, hr_patches)
#             # loss = l1_loss_fn(output, hr_patches)
#             # loss = mse_and_sam_loss(output, label=hr_patches)
#             test_loss += loss.item() * lr_patches.size(0)

#             for j in range(output.size(0)):
#                 sr_img = output[j].detach().cpu().numpy().transpose(1, 2, 0)
#                 hr_img = hr_patches[j].detach().cpu().numpy().transpose(1, 2, 0)

#                 metrics = quality_assessment(hr_img, sr_img, data_range=np.max(hr_img), ratio=2, multi_dimension=False)

#                 for key in all_metrics:
#                     all_metrics[key].append(metrics[key])

#     avg_test_loss = test_loss / len(validation_loader.dataset)

#     print(f"Epoch [{i+1}/{epoch}] - Test Loss: {avg_test_loss:.5f}")
#     for key, values in all_metrics.items():
#         avg_val = sum(values) / len(values)
#         print(f"{key}: {avg_val:.4f}")

#     if (i+1) % 5 == 0:
#         # check_point(epoch, model, optimizer, scheduler, running_loss, test_loss, device)
#         check_point(epoch, model, optimizer, running_loss, test_loss, device)
        
# save_model_path = os.path.join(os.path.dirname(__file__), 'model', 'HyperSuperNet.pt')
# torch.save(model.state_dict(), save_model_path)
# #=========================================================================================


# ###################
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = HSISR_new().to(device)

# res_chain = r'/home/hewit_leo/Internship/my_architectures/my_architecture_6/reschain style/reschian _ chikusei_ report/HyperSuperNet.pt'
# # washinton_path = r'/home/hewit_leo/Internship/my_architectures/my_architecture_6/Internship_iitr_final_outputs/DenseNet Style/Washinton DC dataset/HyperSuperNet.pt'
# # pavia_path = r'/home/hewit_leo/Internship/my_architectures/my_architecture_6/dense net/pavia data/HyperSuperNet.pt'
# # chikusei_path = r'/home/hewit_leo/Internship/my_architectures/my_architecture_6/chikusei 37.9/HyperSuperNet.pt'
# state_dict = torch.load(res_chain, map_location=device)
# model.load_state_dict(state_dict)
# model.eval()
# ######################

# testing
checkpoint_path = r'/home/hewit_leo/my_architecture_1/model/ablation/arch_7_washinton dc/HyperSuperNet.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)

model.eval()
test_loss = 0.0

all_metrics = {'PSNR': [], 'SAM': [], 'SSIM': [], 'ERGAS': [],'CrossCorrelation': []}

printed_shapes = False
with torch.no_grad():
    for lr_patches, hr_patches in test_loader:
        if not printed_shapes:
            print('Lr dim:', lr_patches[0].shape)
            print('Hr dim:', hr_patches[0].shape)
            printed_shapes = True
        lr_patches, hr_patches = lr_patches.to(device), hr_patches.to(device)
        output = model(lr_patches)

        for j in range(output.size(0)):
            sr_img = output[j].detach().cpu().numpy().transpose(1, 2, 0)
            hr_img = hr_patches[j].detach().cpu().numpy().transpose(1, 2, 0)

            metrics = quality_assessment(hr_img, sr_img, data_range=np.max(hr_img), ratio=2, multi_dimension=False)

            all_metrics['PSNR'].append(metrics['PSNR'])
            all_metrics['SAM'].append(metrics['SAM'])
            all_metrics['SSIM'].append(metrics['SSIM'])
            all_metrics['ERGAS'].append(metrics['ERGAS'])
            all_metrics['CrossCorrelation'].append(metrics['CrossCorrelation'])


print("\n===== Final Evaluation on Full Test Set =====")
for key, values in all_metrics.items():
    avg_val = sum(values) / len(values)
    print(f"{key}: {avg_val:.4f}")
print(f"Total Training Time: {(time.time() - start):.2f} seconds")
print("=============================================\n")

model.eval()

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

import random
import numpy as np
import matplotlib.pyplot as plt
import torch

band = 48 
num_samples = 6
num_samples = min(num_samples, len(test_dataset))
indices = random.sample(range(len(test_dataset)), num_samples)


##[80, 50, 20] chikusei and [80, 40, 25] pavia centre [80, 30, 17] washinton d fcc

fcc_bands = [80, 30, 17]

def normalize(img):
    img = np.clip(img, 0, None)
    return img / (np.max(img) + 1e-6)

fig, axes = plt.subplots(nrows=num_samples + 1, ncols=4, figsize=(12, (num_samples + 1) * 2),
                         gridspec_kw={'height_ratios': [0.3] + [1] * num_samples,
                                      'width_ratios': [1, 1, 1, 0.7]})
fig.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.4)

#### for washinton dc
def normalize_img(image, bands):
    img = np.stack([image[b] for b in bands], axis=-1)
    # Percentile normalization to reduce outliers’ visual impact
    for c in range(3):
        minv = np.percentile(img[..., c], 1)
        maxv = np.percentile(img[..., c], 99)
        img[..., c] = np.clip((img[..., c] - minv) / (maxv - minv + 1e-6), 0, 1)
    return img
####

col_titles = ["Low Resolution Patch", "Super Resolved Patch", "Ground Truth", "Metrics"]
for col, title in enumerate(col_titles):
    axes[0, col].text(0.5, 0.5, title, fontsize=14, ha='center', va='center')
    axes[0, col].axis('off')  # Hide axis for title row


for i, idx in enumerate(indices):
    lr_patch, hr_patch = test_dataset[idx]
    lr_patch = lr_patch.unsqueeze(0).to(device)
    hr_patch_np = hr_patch.numpy()


    with torch.no_grad():
        pred = model(lr_patch)
    pred = pred[0].cpu().numpy()
    lr_patch_np = lr_patch[0].cpu().numpy()

    # lr_fcc = np.stack([normalize(lr_patch_np[b]) for b in fcc_bands], axis=-1)
    # sr_fcc = np.stack([normalize(pred[b]) for b in fcc_bands], axis=-1)
    # hr_fcc = np.stack([normalize(hr_patch_np[b]) for b in fcc_bands], axis=-1)

    # axes[i + 1, 0].imshow(lr_fcc)
    # axes[i + 1, 1].imshow(sr_fcc)
    # axes[i + 1, 2].imshow(hr_fcc)

    ### for washinton dc
    lr_fcc = normalize_img(lr_patch_np, fcc_bands)
    sr_fcc = normalize_img(pred, fcc_bands)
    hr_fcc = normalize_img(hr_patch_np, fcc_bands)

    axes[i + 1, 0].imshow(lr_fcc)
    axes[i + 1, 1].imshow(sr_fcc)
    axes[i + 1, 2].imshow(hr_fcc)
    ###

    for j in range(3):
        axes[i + 1, j].axis('off')

    from metrics import quality_assessment
    metrics = quality_assessment(hr_patch_np, pred, data_range=np.max(hr_patch_np), ratio=2, multi_dimension=False)

    metrics_text = (
        f"PSNR: {metrics['PSNR']:.2f} dB\n"
        f"SAM: {metrics['SAM']:.2f}\n"
        f"SSIM: {metrics['SSIM']:.3f}\n"
        # f"ERGAS: {metrics['ERGAS']:.2f}\n"
        f"CC: {metrics['CrossCorrelation']:.3f}"
    )

    axes[i + 1, 3].text(0.5, 0.5, metrics_text, fontsize=9, va='center', ha='center',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    axes[i + 1, 3].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("comparison_grid_6samples_FCC_with_metrics_titles.png", dpi=1200)
plt.show()








import os
from datetime import datetime
from contextlib import redirect_stdout
from torchinfo import summary  # pip install torchinfo

# === Info to Log ===
model_name = model.__class__.__name__
num_epochs = epoch
learning_rate = optimizer.param_groups[0]['lr']
num_train_patches = len(train_dataset)
num_val_patches = len(validation_dataset)
num_test_patches = len(test_dataset)
lr_patch_shape = train_dataset[0][0].shape  # [C, H, W]
hr_patch_shape = train_dataset[0][1].shape

# === Create output directory ===
save_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(save_dir, exist_ok=True)
model_summary_path = os.path.join(save_dir, 'model_summary.txt')

# === Write Info ===
with open(model_summary_path, 'a', encoding='utf-8') as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"Model Summary - {datetime.now()}\n")
    f.write("=" * 80 + "\n")
    f.write(f"Model Name       : {model_name}\n")
    f.write(f"Epochs           : {num_epochs}\n")
    f.write(f"Learning Rate    : {learning_rate}\n")
    f.write(f"Train Patches    : {num_train_patches}\n")
    f.write(f"Val Patches      : {num_val_patches}\n")
    f.write(f"Test Patches     : {num_test_patches}\n")
    f.write(f"LR Patch Shape   : {lr_patch_shape}\n")
    f.write(f"HR Patch Shape   : {hr_patch_shape}\n")
    f.write("-" * 80 + "\n")

    try:
        dummy_input = (1, *lr_patch_shape)  # e.g., (1, 128, 32, 32)
        with redirect_stdout(f):
            summary(model, input_size=dummy_input, device=str(device))
    except Exception as e:
        f.write(f"[!] Could not generate model summary: {e}\n")

    f.write("=" * 80 + "\n\n")

print(f"[✓] Model summary and training info saved to: {model_summary_path}")


from datetime import datetime
import os

# === Metric Summary ===
test_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
metric_values = [
    sum(all_metrics['PSNR']) / len(all_metrics['PSNR']),
    sum(all_metrics['SSIM']) / len(all_metrics['SSIM']),
    sum(all_metrics['SAM']) / len(all_metrics['SAM']),
    sum(all_metrics['ERGAS']) / len(all_metrics['ERGAS']),
    sum(all_metrics['CrossCorrelation']) / len(all_metrics['CrossCorrelation']),
]

# === Paths ===
save_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(save_dir, exist_ok=True)
results_path = os.path.join(save_dir, "test_result_log.txt")

# === Format Settings ===
headers = ["Time", "PSNR", "SSIM", "SAM", "ERGAS", "CrossCorrelation"]
header_format = "{:<20} {:>8} {:>8} {:>8} {:>8} {:>18}"
value_format  = "{:<20} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>18.4f}"

# === Write ===
file_exists = os.path.exists(results_path)
with open(results_path, 'a', encoding='utf-8') as f:
    if not file_exists:
        f.write(header_format.format(*headers) + "\n")
        f.write("-" * 78 + "\n")
    f.write(value_format.format(test_time, *metric_values) + "\n")

print(f"[✓] Test results appended to: {results_path}")



######################
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from torch.nn.functional import normalize

# === Get 1 random patch from test dataset ===
sample_idx = random.randint(0, len(test_dataset) - 1)
lr_patch, hr_patch = test_dataset[sample_idx]
lr_patch = lr_patch.unsqueeze(0).to(device)

# === Run prediction ===
with torch.no_grad():
    sr_patch = model(lr_patch)[0].cpu().numpy()  # [C, H, W]

hr_patch = hr_patch.numpy()  # [C, H, W]

# === Pick 5 random pixel positions ===
C, H, W = hr_patch.shape
num_pixels = 5
coords = [(random.randint(0, H - 1), random.randint(0, W - 1)) for _ in range(num_pixels)]

# === Function to compute SAM (in degrees) ===
def compute_sam(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    cos_sim = np.clip(np.dot(a, b), -1.0, 1.0)
    angle_rad = np.arccos(cos_sim)
    return np.degrees(angle_rad)

# === Plot the spectral curves ===
fig, axes = plt.subplots(nrows=num_pixels, ncols=1, figsize=(8, 10), sharex=True)

for idx, (i, j) in enumerate(coords):
    gt_spectrum = hr_patch[:, i, j]
    pred_spectrum = sr_patch[:, i, j]

    sam_value = compute_sam(gt_spectrum, pred_spectrum)

    axes[idx].plot(gt_spectrum, label="Ground Truth", color='blue')
    axes[idx].plot(pred_spectrum, label="Predicted", color='red')
    axes[idx].set_title(f"Pixel ({i}, {j}) | SAM: {sam_value:.2f}°")
    axes[idx].set_ylabel("Spectral Value")
    axes[idx].grid(True)

axes[0].legend(loc='upper right')
axes[-1].set_xlabel("Bands")
plt.suptitle("Spectral Comparison of 5 Random Pixels", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("spectral_curves_5pixels.png", dpi=300)
plt.show()
