import pandas as pd
import os
import nibabel as nib
import numpy as np
import torch
import re
from torch.utils.data import Dataset

class HippocampusDataset3D(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        """
        Initialise le dataset en séparant les hippocampes.
        :param data_dir: Répertoire contenant les fichiers NIfTI.
        :param csv_path: Chemin vers le fichier CSV contenant les métadonnées.
        :param transform: Transformations optionnelles pour les images.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Charger le CSV
        self.metadata = pd.read_csv(csv_path, header=None)
        
        self.metadata.columns = [
            'subject_id', 'patient_id', 'age', 'gender', 'diagnosis', 
            'apoe', 'education', 'mmse', 'clinical_status', 'label1', 'label2'
        ]
        
        self.metadata['binary_label'] = self.metadata['diagnosis'].apply(lambda x: 1 if x in ['AD', 'MCI'] else 0)
        
        self.image_files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith('n_mmni_') and f.endswith('.nii.gz')]
        )
        
        self.mask_files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith('mask_n_mmni_') and f.endswith('.nii.gz')]
        )
        
        self.image_ids = [re.search(r'(\d{3}_S_\d{4})', f).group(1) for f in self.image_files]
        
        self.coords = [
            (slice(40, 80), slice(90, 130), slice(40, 80)),     # Hippocampe droit
            (slice(100, 140), slice(90, 130), slice(40, 80))    # Hippocampe gauche
        ]
        
    def __len__(self):
        return len(self.image_files) * 2  # 2 hippocampes par image
    
    def __getitem__(self, idx):
        original_idx = idx // 2
        hippo_position = idx % 2
        
        img_path = os.path.join(self.data_dir, self.image_files[original_idx])
        mask_path = os.path.join(self.data_dir, self.mask_files[original_idx])
        
        subject_id = re.search(r'(\d{3}_S_\d{4})', self.image_files[original_idx]).group(1)
        
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)
        
        image = img_nii.get_fdata()
        mask = mask_nii.get_fdata()
        
        c = self.coords[hippo_position]
        
        image_patch = np.ascontiguousarray(image[c])
        mask_patch = np.ascontiguousarray(mask[c])
        
        filtered_metadata = self.metadata[self.metadata['subject_id'] == subject_id]
        if not filtered_metadata.empty:
            label = filtered_metadata['binary_label'].values[0]
        else:
            print(f"Warning: No metadata found for subject ID: {subject_id}")
            label = 0  # Label par défaut
        
        image_tensor = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_patch, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, mask_tensor, label_tensor