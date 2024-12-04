import pandas as pd
import os
import nibabel as nib
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize

transform = Compose([
    lambda x: x / 255.0,  
    Normalize(mean=[0.5], std=[0.5])
])

class HippocampusDataset3D(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Charger et filtrer le CSV
        self.metadata = pd.read_csv(csv_path, header=None)
        self.metadata.columns = [
            'subject_id', 'patient_id', 'age', 'gender', 'diagnosis', 
            'apoe', 'education', 'mmse', 'clinical_status', 'label1', 'label2'
        ]

        # Filtrer uniquement CN et AD
        self.metadata = self.metadata[self.metadata['diagnosis'].isin(['CN', 'AD'])]

        # Créer un label binaire : 0 pour CN, 1 pour AD
        self.metadata['binary_label'] = self.metadata['diagnosis'].apply(lambda x: 1 if x == 'AD' else 0)

        print(self.metadata)

        # Charger les fichiers d'images et de masques, et filtrer par sujet ID
        self.image_files = []
        self.mask_files = []
        for f in os.listdir(data_dir):
            if f.startswith('n_mmni_') and f.endswith('.nii.gz'):
                subject_id = re.search(r'(\d{3}_S_\d{4})', f).group(1)
                if subject_id in self.metadata['subject_id'].values:
                    self.image_files.append(f)
            elif f.startswith('mask_n_mmni_') and f.endswith('.nii.gz'):
                subject_id = re.search(r'(\d{3}_S_\d{4})', f).group(1)
                if subject_id in self.metadata['subject_id'].values:
                    self.mask_files.append(f)

        # Trier les fichiers pour garantir un ordre cohérent
        self.image_files = sorted(self.image_files)
        self.mask_files = sorted(self.mask_files)

        self.coords = [
            (slice(40, 80), slice(90, 130), slice(40, 80)),     # Hippocampe droit
            (slice(100, 140), slice(90, 130), slice(40, 80))    # Hippocampe gauche
        ]
    
    def __len__(self):
        return len(self.image_files) * 2

    def __getitem__(self, idx):
        original_idx = idx // 2  # Identifier l'image originale
        hippo_position = idx % 2  # Identifier l'hippocampe (droit ou gauche)

        img_path = os.path.join(self.data_dir, self.image_files[original_idx])
        mask_path = os.path.join(self.data_dir, self.mask_files[original_idx])

        # Extraire le subject_id à partir du nom du fichier
        subject_id = re.search(r'(\d{3}_S_\d{4})', self.image_files[original_idx]).group(1)

        # Charger les fichiers NIfTI
        img_nii = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        image = img_nii.get_fdata()
        mask = mask_nii.get_fdata()

        # Extraire le patch de l'hippocampe en fonction de la position
        c = self.coords[hippo_position]
        image_patch = np.copy(image[c])  # Créer une copie explicite de l'array
        mask_patch = np.copy(mask[c])    # Créer une copie explicite de l'array

        if hippo_position == 0:  # Hippocampe droit
            image_patch = np.flip(image_patch, axis=0)
            mask_patch = np.flip(mask_patch, axis=0)
                

        # Récupérer le label depuis les métadonnées
        filtered_metadata = self.metadata[self.metadata['subject_id'] == subject_id]
        if not filtered_metadata.empty:
            label = filtered_metadata['binary_label'].values[0]
        else:
            print(f"Warning: No metadata found for subject ID: {subject_id}")
            label = 0  

        # Convertir en tenseurs
        image_tensor = torch.from_numpy(image_patch.copy()).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_patch.copy()).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Appliquer la transformation (normalisation)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor, label_tensor
