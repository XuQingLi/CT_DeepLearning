import os
import Config
import numpy as np
import pydicom
import nibabel as nib
from PIL import Image

CT_data_path =Config.CT_data_dir
root_path=Config.root_dir
# 加载并查看文件格式
def load_ct_data(CT_data_path):
    # 遍历CT_data_path下的所有文件夹
    for folder in os.listdir(CT_data_path):
        folder_path = os.path.join(CT_data_path, folder)
        if os.path.isdir(folder_path):
            # 尝试读取nii文件
            nii_path = os.path.join(CT_data_path, folder + '.nii')
            if os.path.exists(nii_path):
                nii_image = nib.load(nii_path)
                # 获取头部信息
                header = nii_image.header
                print(header)
                # 获取数据
                data = nii_image.get_fdata()
                print(data.shape)
            # 遍历同一文件夹下的DICOM文件
            dcm_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
            for dcm_file in dcm_files:
                dcm_path = os.path.join(folder_path, dcm_file)
                dcm_image = pydicom.dcmread(dcm_path)
                print(f'Loaded DICOM image from {dcm_path}')
                print(f"Patient ID: {dcm_image.PatientID}")
                print(f"Study Date: {dcm_image.StudyDate}")
                # 获取图像数据
                data = dcm_image.pixel_array
                print(data.shape)

def data_visual(CT_data_path):
    # 创建输出文件夹
    output_folder = os.path.join(root_path, 'CT_visual')
    os.makedirs(output_folder, exist_ok=True)
    # 遍历 CT_data_path 下的所有文件夹
    for folder in os.listdir(CT_data_path):
        folder_path = os.path.join(CT_data_path, folder)
        if os.path.isdir(folder_path):
            # 1) 处理 NIfTI 文件
            # 假设与文件夹同名的 .nii 文件保存在 CT_data_path 下，而不是 folder_path 下
            nii_path = os.path.join(CT_data_path, folder + '.nii')
            if os.path.exists(nii_path):
                sub_folder = os.path.join(output_folder, folder)
                os.makedirs(sub_folder, exist_ok=True)
                # 1) 加载 NIfTI 文件
                nii_image = nib.load(nii_path)
                data = nii_image.get_fdata()  # shape ~ (512, 512, 71)，类型 float

                # 2) 计算全局 min、max（避免单切片没有对比度）
                global_min = data.min()
                global_max = data.max()

                # 如果 global_max == global_min，则说明整幅 3D 图像都是同一个值
                # 这种情况只能输出全黑（或者说一个纯色图）
                if global_max == global_min:
                    # 直接全部设为 0
                    data_normalized = np.zeros_like(data)
                else:
                    # 进行全局 min-max 归一化
                    data_normalized = (data - global_min) / (global_max - global_min)

                # 3) 遍历 3D 第三维（切片）并输出 BMP
                depth = data_normalized.shape[2]
                for z in range(depth):
                    slice_data = data_normalized[:, :, z]

                    # 将 [0,1] 转为 [0,255] 的 8 位灰度
                    slice_data = (slice_data * 255).astype(np.uint8)

                    # 利用 PIL 保存为 BMP
                    img = Image.fromarray(slice_data)
                    bmp_name = f"{folder}_{z:03d}.bmp"
                    output_path = os.path.join(sub_folder, bmp_name)
                    img.save(output_path)

            # 2) 处理 DICOM 文件
            dcm_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
            for dcm_file in dcm_files:
                dcm_path = os.path.join(folder_path, dcm_file)
                dcm_image = pydicom.dcmread(dcm_path)
                pixel_array = dcm_image.pixel_array  # 一般为2D

                # 同样进行 0-255 的归一化处理
                px_min, px_max = pixel_array.min(), pixel_array.max()
                if px_max > px_min:
                    pixel_array = (pixel_array - px_min) / (px_max - px_min)
                else:
                    pixel_array = np.zeros_like(pixel_array)
                pixel_array = (pixel_array * 255).astype(np.uint8)

                # 保存为 BMP
                img = Image.fromarray(pixel_array)
                # 原名去掉 .dcm 后缀，改为 .bmp
                bmp_name = os.path.splitext(dcm_file)[0] + ".bmp"
                output_path = os.path.join(sub_folder, bmp_name)
                img.save(output_path)


# load_ct_data(CT_data_path)
data_visual(CT_data_path)