import os
import nibabel as nib
import numpy as np
import Config

CT_nifti_dir =Config.CT_nifti_dir 

# 遍历文件夹下所有文件
for file_name in os.listdir(CT_nifti_dir):
    # 只处理 .nii.gz 的原始影像文件
    if file_name.endswith(".nii.gz"):
        # 得到患者ID（去掉 .nii.gz 后缀）
        patient_id = file_name[:-7]  # 或 replace(".nii.gz", "")

        # 构造标注文件名（假设标注后缀是 .nii）
        label_name = patient_id + ".nii"

        # 拼接得到完整路径
        img_path = os.path.join(CT_nifti_dir, file_name)
        label_path = os.path.join(CT_nifti_dir, label_name)

        # 检查标注文件是否存在
        if not os.path.exists(label_path):
            print(f"[警告] {label_name} 不存在，跳过该患者 {patient_id}。")
            continue

        # 读取 NIfTI 文件
        img_nifti = nib.load(img_path)
        label_nifti = nib.load(label_path)

        # 获取数据与仿射矩阵
        img_data = img_nifti.get_fdata(dtype=np.float32)
        label_data = label_nifti.get_fdata(dtype=np.float32)

        img_affine = img_nifti.affine
        label_affine = label_nifti.affine

        # 检查形状是否一致
        if img_data.shape != label_data.shape:
            print(f"[形状不匹配] 患者 {patient_id}: 原始图像形状 {img_data.shape}, 标注形状 {label_data.shape}")
        else:
            print(f"[形状匹配] 患者 {patient_id}: 形状 {img_data.shape} 一致")

        # 可选：检查仿射矩阵 (affine) 是否一致
        if not np.allclose(img_affine, label_affine, atol=1e-5):
            print(f"[警告] 患者 {patient_id} 的affine不一致：")
            print("原始影像 affine:\n", img_affine)
            print("标注 affine:\n", label_affine)
        else:
            print(f"[仿射匹配] 患者 {patient_id} 的affine一致")

        print("-----------------------------------")
