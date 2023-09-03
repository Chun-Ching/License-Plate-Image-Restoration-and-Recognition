# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# test_img_folder = 'LR/*'

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# print('Model path {:s}. \nTesting...'.format(model_path))

# idx = 0
# for path in glob.glob(test_img_folder):
#     idx += 1
#     base = osp.splitext(osp.basename(path))[0]
#     print(idx, base)
#     # read images
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img = img * 1.0 / 255
#     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#     img_LR = img.unsqueeze(0)
#     img_LR = img_LR.to(device)

#     with torch.no_grad():
#         output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#     output = (output * 255.0).round()
#     cv2.imwrite('results/{:s}_rlt.png'.format(base), output)


# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import re

# def numerical_sort(value):
#     # 提取文件名中的数字部分
#     numbers = re.findall(r'\d+', value)
#     if numbers:
#         # 将数字部分转换为整数进行排序
#         return int(numbers[0])
#     else:
#         return value

# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# # device = torch.device('cpu')

# test_img_folder = 'LR_AC_res_2000/*'

# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)

# print('Model path {:s}. \nTesting...'.format(model_path))

# count = 1  # 计数器变量

# file_list = sorted(glob.glob(test_img_folder), key=numerical_sort)

# for path in file_list:
#     base = osp.splitext(osp.basename(path))[0]
#     print(count, base)
#     # read images
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img = img * 1.0 / 255
#     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#     img_LR = img.unsqueeze(0)
#     img_LR = img_LR.to(device)

#     with torch.no_grad():
#         output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#     output = (output * 255.0).round()

#     # 构建新的文件名
#     new_filename = 'results_AC_res_2000/{:d}.png'.format(count)
#     cv2.imwrite(new_filename, output)
    
#     count += 1  # 增加计数





import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import re
import os
from natsort import natsorted

def numerical_sort(value):
    # 提取文件名中的数字部分
    numbers = re.findall(r'\d+', value)
    if numbers:
        # 将数字部分转换为整数进行排序
        return int(numbers[0])
    else:
        return value

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

input_folder = 'LR/*'
output_folder = 'results/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

count = 1  # 计数器变量

file_list = natsorted(glob.glob(input_folder), key=numerical_sort)

for path in file_list:
    base = osp.splitext(osp.basename(path))[0]
    print(count, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    # 构建新的文件名
    new_filename = osp.join(output_folder, '{:d}.png'.format(count))
    cv2.imwrite(new_filename, output)

    count += 1  # 增加计数

print("處理完成。")






