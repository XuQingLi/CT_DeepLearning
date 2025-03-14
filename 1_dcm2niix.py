import os
import numpy as np
import SimpleITK as sitk
import shutil
import pydicom
import pylibjpeg
pydicom.config.image_handlers = ['pylibjpeg']
import Config

CT_raw_data_dir = Config.CT_data_dir    # 原始 DICOM 数据根目录
CT_nifti_dir = Config.CT_nifti_dir      # 输出 NIfTI 的目标目录

def read_dicom_series(dicom_dir):
    """
    从 dicom_dir 读取一套 DICOM 序列（多张切片），并返回对应的:
      1. 3D numpy array: shape = [num_slices, height, width]
      2. spacing: (z_spacing, y_spacing, x_spacing)
      3. origin: (x0, y0, z0)
      4. direction: (传给 SimpleITK 的方向矩阵，长度 9 或 3x3 矩阵展平)
    """
    # 收集本目录下所有 .dcm 或 .dicom 文件的路径
    dicom_files = []
    for root, _, files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith(('.dcm', '.dicom')):
                dicom_files.append(os.path.join(root, f))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    
    # 读取并解析所有切片
    slices = []
    for dcm_file in dicom_files:
        ds = pydicom.dcmread(dcm_file)
        # 若 ds.pixel_array 解码失败时，会抛出异常；确保已安装 pylibjpeg + pylibjpeg-libjpeg
        if not hasattr(ds, 'pixel_array'):
            continue  # 若没像素数据，则跳过
        slices.append(ds)
    
    # 按 InstanceNumber 或者 SliceLocation 排序，以保证序列的空间顺序正确
    slices = sorted(slices, key=lambda s: int(s.InstanceNumber))

    # 提取像素间距（PixelSpacing）等常用元数据
    first_slice = slices[0]
    pixel_spacing = first_slice.PixelSpacing  # [row_spacing, col_spacing]
    x_spacing = float(pixel_spacing[1])
    y_spacing = float(pixel_spacing[0])
    
    # z_spacing 如果没有 SliceThickness，可用相邻两张切片位置差
    if hasattr(first_slice, 'SliceThickness'):
        z_spacing = float(first_slice.SliceThickness)
    else:
        if len(slices) > 1:
            z0 = float(slices[0].ImagePositionPatient[2])
            z1 = float(slices[1].ImagePositionPatient[2])
            z_spacing = abs(z1 - z0)
        else:
            z_spacing = 1.0
    
    # 组装 3D 数组：shape = [num_slices, height, width]
    volume_3d = []
    for s in slices:
        arr_2d = s.pixel_array  # 这里会自动使用 pylibjpeg 进行解码
        volume_3d.append(arr_2d)
    volume_3d = np.stack(volume_3d, axis=0)
    
    # SimpleITK 里约定的 spacing 是 (x_spacing, y_spacing, z_spacing)
    spacing = (x_spacing, y_spacing, z_spacing)

    # 原点 Origin
    if hasattr(first_slice, 'ImagePositionPatient'):
        ipp = first_slice.ImagePositionPatient
        origin = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    else:
        origin = (0.0, 0.0, 0.0)
    
    # 方向 Direction
    if hasattr(first_slice, 'ImageOrientationPatient'):
        iop = first_slice.ImageOrientationPatient
        row_dir = np.array(iop[0:3], dtype=np.float64)
        col_dir = np.array(iop[3:6], dtype=np.float64)
        slice_dir = np.cross(row_dir, col_dir)
        
        direction_matrix = np.array([
            [row_dir[0], col_dir[0], slice_dir[0]],
            [row_dir[1], col_dir[1], slice_dir[1]],
            [row_dir[2], col_dir[2], slice_dir[2]],
        ], dtype=np.float64)
        direction = tuple(direction_matrix.flatten())
    else:
        direction = (1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0)
    
    return volume_3d, spacing, origin, direction

def convert_dicom_folder_to_nifti(dicom_folder, output_nii_path):
    """
    将 dicom_folder 中的一套 DICOM 序列转换为 NIfTI，并写出到 output_nii_path。
    若成功，返回 True；否则返回 False。
    """
    try:
        volume_3d, spacing, origin, direction = read_dicom_series(dicom_folder)
        # 转为 SimpleITK 图像
        sitk_image = sitk.GetImageFromArray(volume_3d)
        # 配置元数据
        sitk_image.SetSpacing(spacing)
        sitk_image.SetOrigin(origin)
        sitk_image.SetDirection(direction)
        
        # 写出 NIfTI
        sitk.WriteImage(sitk_image, output_nii_path)
        return True
    except Exception as e:
        print(f"Failed to convert {dicom_folder} -> {output_nii_path}, error: {e}")
        return False

def convert_all_dicom_in_dir_to_nifti(CT_raw_data_dir, CT_nifti_dir):
    """
    批量遍历 CT_raw_data_dir 下所有子文件夹，尝试将其中的 DICOM 序列转换为 .nii.gz。
    如果遇到 .nii 或 .nii.gz 文件也可直接复制。
    """
    os.makedirs(CT_nifti_dir, exist_ok=True)

    for item in os.listdir(CT_raw_data_dir):
        src_path = os.path.join(CT_raw_data_dir, item)
        
        # 若是 NIfTI 文件，直接复制
        if os.path.isfile(src_path) and item.lower().endswith(('.nii', '.nii.gz')):
            dest_path = os.path.join(CT_nifti_dir, item)
            shutil.copy(src_path, dest_path)
            print(f"Copied NIfTI file: {src_path} -> {dest_path}")
        
        # 若是文件夹，则尝试将该文件夹中的 DICOM 序列转换成 NIfTI
        elif os.path.isdir(src_path):
            output_nii_name = item + '.nii.gz'
            output_nii_path = os.path.join(CT_nifti_dir, output_nii_name)
            ok = convert_dicom_folder_to_nifti(src_path, output_nii_path)
            if ok:
                print(f"Converted DICOM folder: {src_path} -> {output_nii_path}")

if __name__ == "__main__":
    convert_all_dicom_in_dir_to_nifti(CT_raw_data_dir, CT_nifti_dir)
