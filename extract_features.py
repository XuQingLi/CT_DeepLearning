import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from tqdm import tqdm
import Config

root_dir = Config.root_dir
CT_preprocessed=Config.CT_preprocessed
IMAGE_DIR = os.path.join(CT_preprocessed, "images/npy/")
MASK_DIR = os.path.join(CT_preprocessed, "masks/npy/")
FEATURE_DIR=os.path.join(root_dir,"CT_feature")

class CTSliceDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patients = [f.split('.npy')[0] for f in os.listdir(image_dir)]
        self.transform = transform

        self.slices = []  # 保存所有切片信息

        # 加载数据并处理切片
        for patient in tqdm(self.patients, desc='Loading data'):
            img = np.load(os.path.join(image_dir, f'{patient}.npy'))
            mask = np.load(os.path.join(mask_dir, f'{patient}.npy'))

            num_slices = img.shape[0]

            for idx in range(num_slices):
                img_slice = img[idx, :, :]
                mask_slice = mask[idx, :, :]

                # 记录此切片是否含有病灶
                has_lesion = int(mask_slice.sum() > 0)

                self.slices.append({
                    'patient': patient,
                    'slice_idx': idx,
                    'img_slice': img_slice,
                    'mask_slice': mask_slice,
                    'has_lesion': has_lesion
                })

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        slice_data = self.slices[index]
        img = slice_data['img_slice']
        
        # 将单通道的灰度切片扩展为3通道，以适配ResNet18
        img = np.stack([img, img, img], axis=0).astype(np.float32)

        if self.transform:
            img = self.transform(torch.from_numpy(img))

        return {
            'img': img,
            'has_lesion': slice_data['has_lesion'],
            'patient': slice_data['patient'],
            'slice_idx': slice_data['slice_idx']
        }

# 特征提取函数
def extract_features(dataloader, model, device):
    model.eval()
    features = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            imgs = batch['img'].to(device)
            output = model(imgs)  # 提取特征，默认输出特征向量

            # 存储特征
            batch_size = imgs.size(0)
            for i in range(batch_size):
                patient = batch['patient'][i]
                slice_idx = batch['slice_idx'][i]
                has_lesion = batch['has_lesion'][i].item()

                if patient not in features:
                    features[patient] = {'lesion': [], 'non_lesion': []}

                # 根据是否含病灶，分别存储特征
                if has_lesion:
                    features[patient]['lesion'].append(output[i].cpu().numpy())
                else:
                    features[patient]['non_lesion'].append(output[i].cpu().numpy())

    return features

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保特征保存目录存在
    os.makedirs(FEATURE_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = CTSliceDataset(IMAGE_DIR, MASK_DIR, transform)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)


    resnet = resnet18(weights=None)  # 不预加载以规避兼容问题
    weights = ResNet18_Weights.DEFAULT
    state_dict = weights.get_state_dict(progress=True)
    resnet.load_state_dict(state_dict, strict=True)

    resnet.fc = nn.Identity()
    resnet.to(device)

    patient_features = extract_features(dataloader, resnet, device)

    for patient, feats in patient_features.items():
        lesion_feats = np.array(feats['lesion']) if feats['lesion'] else np.zeros((0, 512))
        non_lesion_feats = np.array(feats['non_lesion']) if feats['non_lesion'] else np.zeros((0, 512))

        lesion_feature_mean = lesion_feats.mean(axis=0) if len(lesion_feats) > 0 else np.zeros(512)
        non_lesion_feature_mean = non_lesion_feats.mean(axis=0) if len(non_lesion_feats) > 0 else np.zeros(512)
        patient_feature = np.concatenate([lesion_feature_mean, non_lesion_feature_mean])

        np.save(os.path.join(FEATURE_DIR, f"{patient}.npy"), patient_feature)
        print(f"Saved {patient} features to {FEATURE_DIR}")

if __name__ == '__main__':
    main()
