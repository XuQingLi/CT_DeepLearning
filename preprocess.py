import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import SimpleITK as sitk
import Config

def preprocess_ct_images(ct_dir, output_dir, target_spacing=(1.0, 1.0, 1.0)):
    """
    读取CT图像和对应的掩码，进行归一化和重采样，
    并同时将处理后的结果保存为NIfTI和NumPy(.npy)格式。
    """
    # 创建输出目录结构
    images_nii_dir = os.path.join(output_dir, 'images', 'nii')
    images_npy_dir = os.path.join(output_dir, 'images', 'npy')
    masks_nii_dir = os.path.join(output_dir, 'masks', 'nii')
    masks_npy_dir = os.path.join(output_dir, 'masks', 'npy')
    
    os.makedirs(images_nii_dir, exist_ok=True)
    os.makedirs(images_npy_dir, exist_ok=True)
    os.makedirs(masks_nii_dir, exist_ok=True)
    os.makedirs(masks_npy_dir, exist_ok=True)
    
    # 获取所有CT图像文件
    ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')]
    
    for ct_file in ct_files:
        patient_id = ct_file.replace(".nii.gz", "")  # 获取患者ID
        mask_file = f"{patient_id}.nii"  # 对应的掩码文件
        
        # 检查掩码文件是否存在
        if not os.path.exists(os.path.join(ct_dir, mask_file)):
            print(f"警告: 未找到患者 {patient_id} 的掩码文件")
            continue
        
        print(f"处理患者: {patient_id}")
        
        # 读取CT图像
        ct_path = os.path.join(ct_dir, ct_file)
        ct_img = nib.load(ct_path)
        ct_data = ct_img.get_fdata()
        original_spacing = ct_img.header.get_zooms()
        
        # 读取掩码
        mask_path = os.path.join(ct_dir, mask_file)
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        
        # 归一化CT值 (简单示例：限制到[-1000,400]后归一化到[0,1])
        ct_data_normalized = normalize_ct(ct_data)
        
        # 重采样CT图像和掩码
        ct_resampled, mask_resampled = resample_ct_and_mask(
            ct_data_normalized, 
            mask_data, 
            original_spacing, 
            target_spacing
        )
        
        # === 1) 保存处理后的图像和掩码 (NIfTI) ===
        save_nifti(
            ct_resampled, 
            os.path.join(images_nii_dir, f"{patient_id}.nii.gz"),
            target_spacing
        )
        
        save_nifti(
            mask_resampled, 
            os.path.join(masks_nii_dir, f"{patient_id}.nii.gz"),
            target_spacing
        )

        # === 2) 额外保存为NumPy (.npy) 文件 ===
        np.save(
            os.path.join(images_npy_dir, f"{patient_id}.npy"),
            ct_resampled
        )
        np.save(
            os.path.join(masks_npy_dir, f"{patient_id}.npy"),
            mask_resampled
        )
        
        print(f"患者 {patient_id} 处理完成")
        

def normalize_ct(ct_data, min_bound=-1000, max_bound=400):
    """
    CT值归一化
    
    参数:
    ct_data: CT图像数据
    min_bound: CT值下限，默认为-1000 (HU)
    max_bound: CT值上限，默认为400 (HU)
    
    返回:
    归一化后的CT数据，范围[0,1]
    """
    # 将CT值限制在指定范围内
    ct_data = np.clip(ct_data, min_bound, max_bound)
    # 归一化到[0,1]
    ct_data = (ct_data - min_bound) / (max_bound - min_bound)
    return ct_data

def resample_ct_and_mask(ct_data, mask_data, original_spacing, target_spacing):
    """
    重采样CT图像和掩码到目标分辨率
    
    参数:
    ct_data: CT图像数据
    mask_data: 掩码数据
    original_spacing: 原始体素间距
    target_spacing: 目标体素间距
    
    返回:
    重采样后的CT图像和掩码
    """
    # 计算缩放因子
    resize_factor = [orig / target for orig, target in zip(original_spacing, target_spacing)]
    new_shape = [int(np.round(size * factor)) for size, factor in zip(ct_data.shape, resize_factor)]
    
    # 重采样CT图像 (使用线性插值)
    ct_resampled = ndimage.zoom(ct_data, resize_factor, order=1)
    
    # 重采样掩码 (使用最近邻插值以保持标签值)
    mask_resampled = ndimage.zoom(mask_data, resize_factor, order=0)
    
    return ct_resampled, mask_resampled

def save_nifti(data, output_path, spacing):
    """
    保存数据为NIfTI格式
    
    参数:
    data: 要保存的数据
    output_path: 输出路径
    spacing: 体素间距
    """
    # 创建新的NIfTI图像
    img = nib.Nifti1Image(data, np.eye(4))
    
    # 设置体素间距
    img.header.set_zooms(spacing)
    
    # 保存图像
    nib.save(img, output_path)

if __name__ == "__main__":
    # 设置路径
    CT_nifti_dir = Config.CT_nifti_dir   
    output_dir = os.path.join(Config.root_dir, "CT_preprocessed")   
    
    # 执行预处理并保存为 NIfTI 和 .npy
    preprocess_ct_images(CT_nifti_dir, output_dir)