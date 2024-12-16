import pandas as pd
import os
import nibabel as nib
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize
import h5py

transform = Compose([
    lambda x: x / 255.0,  
    Normalize(mean=[0.5], std=[0.5])
])

class HippocampusDataset3D(Dataset):
    def __init__(self, data_dir, csv_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.metadata = pd.read_csv(csv_path, header=None)
        self.metadata.columns = [
            'subject_id', 'patient_id', 'age', 'gender', 'diagnosis', 
            'apoe', 'education', 'mmse', 'clinical_status', 'label1', 'label2'
        ]

        self.metadata = self.metadata[self.metadata['diagnosis'].isin(['CN', 'AD'])]

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

class MixupAugmentedDataset(Dataset):
    def __init__(self, base_dataset, augment_factor=5, lam_values=None):
        """
        Dataset pour appliquer la méthode Mixup avec des valeurs discrètes de lambda.
        :param base_dataset: Instance du dataset original.
        :param augment_factor: Nombre de fois que chaque donnée est augmentée.
        :param lam_values: Liste des valeurs de lambda discrètes.
        """
        self.base_dataset = base_dataset
        self.augment_factor = augment_factor
        self.lam_values = torch.tensor([i / 10 for i in range(1, 11)])  # [0.1, 0.2, ..., 1.0]
        
        self.augmented_indices = self._generate_augmentation_indices()

    def _generate_augmentation_indices(self):
        indices_pairs = []
        base_len = len(self.base_dataset)
        
        for _ in range(base_len * self.augment_factor):
            idx1 = np.random.randint(0, base_len)
            idx2 = np.random.randint(0, base_len)
            indices_pairs.append((idx1, idx2))
        print(indices_pairs)
        
        return indices_pairs

    def __len__(self):
        return len(self.base_dataset) * (1 + self.augment_factor)

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            print(idx)
            return self.base_dataset[idx]
        else:
            print(idx)
            aug_idx = idx - len(self.base_dataset)
            idx1, idx2 = self.augmented_indices[aug_idx]
            
            img1, mask1, label1 = self.base_dataset[idx1]
            img2, mask2, label2 = self.base_dataset[idx2]
            
            lam = np.random.choice(self.lam_values.numpy())
            
            mixed_image = lam * img1 + (1 - lam) * img2
            mixed_label = lam * label1 + (1 - lam) * label2
            
            return mixed_image, mask1, mixed_label

    def generate_and_save_dataset(self, output_path):
        """
        Génère le dataset avec Mixup et sauvegarde dans un fichier HDF5.
        :param output_path: Chemin pour sauvegarder le dataset.
        """
        base_len = len(self.base_dataset)
        total_len = len(self)
        
        # Créer un fichier HDF5
        with h5py.File(output_path, 'w') as f:
            # Préallocation des datasets
            images = f.create_dataset('images', shape=(total_len, *self.base_dataset[0][0].shape), dtype='f')
            masks = f.create_dataset('masks', shape=(total_len, *self.base_dataset[0][1].shape), dtype='f')
            labels = f.create_dataset('labels', shape=(total_len,), dtype='f')

            # Sauvegarde des données originales
            for idx in range(base_len):
                print(idx)
                img, mask, label = self.base_dataset[idx]
                images[idx] = img
                masks[idx] = mask
                labels[idx] = label

            # Sauvegarde des données augmentées (Mixup)
            for aug_idx in range(len(self.augmented_indices)):
                print(aug_idx)
                idx1, idx2 = self.augmented_indices[aug_idx]
                img1, mask1, label1 = self.base_dataset[idx1]
                img2, _, label2 = self.base_dataset[idx2]
                lam = np.random.choice(self.lam_values.numpy())
                mixed_image = lam * img1 + (1 - lam) * img2
                mixed_label = lam * label1 + (1 - lam) * label2

                images[base_len + aug_idx] = mixed_image
                masks[base_len + aug_idx] = mask1
                labels[base_len + aug_idx] = mixed_label

        print(f"Dataset Mixup sauvegardé dans {output_path}")

class TestHippocampusDataset3D(Dataset):
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
        self.metadata = self.metadata[self.metadata['diagnosis'].isin(['MCI'])]
        self.metadata['apoe'] = self.metadata['apoe'].astype(int)
        self.metadata = self.metadata[self.metadata['apoe'].isin([3, 4])]


        # Créer un label binaire : 0 pour CN, 1 pour AD
        self.metadata['binary_label'] = self.metadata['apoe'].apply(lambda x: 0 if x == 4 else 1)

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

        return image_tensor, label_tensor

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        """
        Initialise le dataset à partir d'un fichier HDF5.
        :param h5_file_path: Chemin vers le fichier .h5.
        """
        self.h5_file_path = h5_file_path

        # Ouvrir le fichier une première fois pour obtenir la longueur des données
        with h5py.File(self.h5_file_path, 'r') as f:
            self.data_len = len(f['images'])  # Longueur des données (nombre d'images)
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # Ouvrir le fichier au moment de l'accès pour éviter de le garder ouvert
        with h5py.File(self.h5_file_path, 'r') as f:
            image = torch.tensor(f['images'][idx], dtype=torch.float32)
            label = torch.tensor(f['labels'][idx], dtype=torch.float32).unsqueeze(0)
        
        return image, label