#%%
# 从原始数据中找出所有的CT序列，并存入新的文件夹
import os
import csv
import shutil
from tqdm import tqdm

data_dir = "D:\\DICOM\\manifest-1600709154662\\LIDC-IDRI"
output_dir = "D:\\DICOM\\manifest-1600709154662\\Raw_data"
csv_path = "D:\\DICOM\\manifest-1600709154662\\data.csv"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

data_list = []
with open(csv_path, 'w', newline="") as f:
    writer = csv.writer(f)
    for maindir, subdir, file_name_list in os.walk(data_dir):
        if len(file_name_list) > 50:
            data_list.append(maindir)
print(os.listdir(data_list[0]))

file_count = 0
for item in tqdm(data_list):
    file_dir = output_dir + "\\" + str(file_count)
    shutil.copytree(item, file_dir, ignore=shutil.ignore_patterns('*.xml'))
    file_count += 1
#%%
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import pydicom
import os
import dicom2nifti.patch_pydicom_encodings
from tqdm import tqdm
import csv

raw_ct_dir = "D:/DICOM/manifest-1600709154662/Raw_data/"
raw_list = os.listdir(raw_ct_dir)
output = "D:/DICOM/manifest-1600709154662/ct_nii/"


# resample to 128*128
def ImageResampleSize(sitk_image, img_size=128, is_label=False):
    # 获取原来CT的Size和Spacing
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())

    new_size = (img_size, img_size, size[2])
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)
    newimage = resample.Execute(sitk_image)
    return newimage


def saved_preprocessed(savedImg, origin, direction, xyz_thickness):
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
    return newImg


with open('D:/DICOM/manifest-1600709154662/img_info.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(
    ["file_name", "origin_spacing", "origin_Origin", "origin_Direction", "origin_Size",
     "resize_spacing", "resize_Origin", "resize_Direction", "resize_Size"])

    for file in tqdm(raw_list):
        # 读取dicom序列
        file_path = os.path.join(raw_ct_dir, file)
        print(f'file_path:{file_path}')
        output_path = output + file + ".nii"

        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()  # 这一步是加载公开的元信息
        reader.LoadPrivateTagsOn()  # 这一步是加载私有的元信息
        img_names = reader.GetGDCMSeriesFileNames(file_path)

        reader.SetFileNames(img_names)
        image = reader.Execute()
        # 原来的info
        origin_spacing = image.GetSpacing()
        origin_Origin = image.GetOrigin()
        origin_Direction = image.GetDirection()
        origin_Size = image.GetSize()

        normal = ImageResampleSize(image)
        # resize后的Info
        resize_spacing = normal.GetSpacing()
        resize_Origin = normal.GetOrigin()
        resize_Direction = normal.GetDirection()
        resize_Size = normal.GetSize()

        # 写入信息
        csv_writer.writerow([file, origin_spacing, origin_Origin, origin_Direction, origin_Size, resize_spacing,
                            resize_Origin, resize_Direction, resize_Size])

        # 获取CT_array
        np_arr = sitk.GetArrayFromImage(normal)
        # 去除CT边界图像
        np_arr[np_arr < -500] = -500
        # 将CT修正为128*128*128
        x, y, z = np_arr.shape
        if z < x or z < y:
            startx = (x - z) // 2
            starty = (y - z) // 2
            # 裁剪成一个正方体
            cubed_arr = np_arr[startx:startx + z, starty:starty + z, :]
        else:
            max_xyz = max(x, y, z)
            x_pre_pad = (max_xyz - x) // 2
            # (8-6)//2=1
            x_post_pad = max_xyz - x - x_pre_pad
            # 8-6-1=1
            y_pre_pad = (max_xyz - y) // 2
            y_post_pad = max_xyz - y - y_pre_pad
            # 填充成一个正方体
            cubed_arr = np.pad(np_arr, ((x_pre_pad, x_post_pad), (y_pre_pad, y_post_pad), (0, 0)), mode='constant',
                            constant_values=0)
        print("process-2: resize to cube !  ", cubed_arr.shape)

        x1, y1, z1 = cubed_arr.shape
        assert x1 == y1, 'x and y dimensions are the same size'
        assert y1 == z1, 'y and z dimensions are the same size'
        assert z1 == x1, 'z and x dimensions are the same size'

        # resize_img = sitk.GetImageFromArray(cubed_arr)
        resize_img = saved_preprocessed(cubed_arr, resize_Origin, resize_Direction, resize_spacing)
        print(resize_img.GetSize())
        sitk.WriteImage(resize_img, output_path)
# %%
