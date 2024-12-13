import torch
import torch.optim as optim
import csv
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from engine import train_one_epoch, validate
from dataset import HippocampusDataset3D, TestHippocampusDataset3D, MixupAugmentedDataset
import matplotlib.pyplot as plt
import numpy as np
from utils import EarlyStopping
import os
from datetime import datetime
from itertools import product
from model import Basic3DCNN 
from sklearn.metrics import classification_report

torch.cuda.empty_cache()

# Espaces de recherche pour les hyperparamètres
param_grid = {
    "learning_rate": [0.001],
    "batch_size": [16],
    "weight_decay": [0],
    "num_epochs": [200],  
}

# Combinaisons de tous les hyperparamètres
param_combinations = list(product(*param_grid.values()))

def main():
    # Nom du fichier CSV
    results_file = "hyperparameter_results.csv"

    # Initialiser le fichier CSV avec les en-têtes
    with open(results_file, mode='w', newline='') as file:
        writer2 = csv.writer(file)
        writer2.writerow(["learning_rate", "batch_size", "weight_decay", 
                        "num_epochs", "val_loss", "val_accuracy", "train_loss", "train_accuracy"])

    best_val_accuracy = 0
    best_hyperparams = None

    for params in param_combinations:

        learning_rate, batch_size, weight_decay, num_epochs = params

        # Paramètres
        input_channels = 1
        num_classes = 2
        # batch_size = 32
        # num_epochs = 100
        # learning_rate = 0.001
        data_dir = "/home/ychappetjuan/Alzeimer/datasets"
        csv_path = "/home/ychappetjuan/Alzeimer/list_standardized_tongtong_2017.csv"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = Basic3DCNN(input_channels=input_channels, num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        writer = SummaryWriter(log_dir='./logs') 

        base_dataset = HippocampusDataset3D(data_dir, csv_path)
        dataset = MixupAugmentedDataset(base_dataset, augment_factor=3)

        print("Taille totale du dataset :", len(dataset))

        labels = [dataset[i][2] for i in range(len(dataset))]

        print(labels)

        # image_right, mask_right, label_right = dataset[2]  # Hippocampe droit
        # image_left, mask_left, label_left = dataset[3]    # Hippocampe gauche

        # image_right_np = image_right.squeeze().numpy()
        # image_left_np = image_left.squeeze().numpy()

        # plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # plt.imshow(image_right_np[:, :, 30], cmap='gray')
        # plt.title('Hippocampe Droit (Flipped)')
        # plt.subplot(122)
        # plt.imshow(image_left_np[:, :, 30], cmap='gray')
        # plt.title('Hippocampe Gauche')
        # plt.show()

        binned_labels = np.digitize(labels, bins=[0.25, 0.5, 0.75])

        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=0.2,
            stratify=binned_labels,
            random_state=42
        )

        print("After stratify")

        run_dir = "/home/ychappetjuan/Alzeimer/models"

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
        unique_run_dir = os.path.join(run_dir, current_time)

        os.makedirs(unique_run_dir, exist_ok=True)

        checkpoint_dir = os.path.join(unique_run_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Dossier pour le run créé : {unique_run_dir}")
        print(f"Dossier checkpoints créé : {checkpoint_dir}")

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        early_stopping = EarlyStopping(
            patience=7,
            verbose=True,
            path=os.path.join(checkpoint_dir, 'best_model.pth')
        )

        for epoch in range(num_epochs):
            print(f"Époque {epoch + 1}/{num_epochs}")

            train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Perte d'entraînement: {train_loss:.4f}, Précision d'entraînement: {train_accuracy:.4f}")

            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            print(f"Perte de validation: {val_loss:.4f}, Précision de validation: {val_accuracy:.4f}")

            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        with open(results_file, mode='a', newline='') as file:
            writer2 = csv.writer(file)
            writer2.writerow([learning_rate, batch_size, weight_decay, 
                            num_epochs, val_loss, val_accuracy, train_loss, train_accuracy])
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_hyperparams = (learning_rate, batch_size, weight_decay, num_epochs)

        print(f"Meilleurs hyperparamètres: {best_hyperparams}")
        print(f"Meilleure précision de validation: {best_val_accuracy:.4f}")

        writer.close()

def test():
    # Répertoires et configuration
    data_dir = "/home/ychappetjuan/Alzeimer/datasets"
    csv_path = "/home/ychappetjuan/Alzeimer/list_standardized_tongtong_2017.csv"
    checkpoint_path = "/home/ychappetjuan/Alzeimer/models/2024-12-06_08-42-54/checkpoints/best_model.pth"  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Charger le dataset de test
    test_dataset = TestHippocampusDataset3D(data_dir, csv_path)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Charger le modèle entraîné
    input_channels = 1
    num_classes = 2
    model = Basic3DCNN(input_channels=input_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # Mode évaluation

    all_predictions = []
    all_labels = []

    # Itérer sur les données de test
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Prédictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcul des métriques
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Précision sur le jeu de test : {accuracy:.4f}")
    print(classification_report(all_labels, all_predictions))


if __name__ == "__main__":
    main()
