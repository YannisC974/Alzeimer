import torch
import torch.optim as optim
import csv
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from engine import train_one_epoch, validate, train_one_epoch_mixup, validate_mixup
from dataset import HippocampusDataset3D, TestHippocampusDataset3D, MixupAugmentedDataset
import matplotlib.pyplot as plt
import numpy as np
from utils import EarlyStopping
import os
import optuna
from datetime import datetime
from itertools import product
from model import Basic3DCNN, GradCAM3D
from sklearn.metrics import classification_report
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

torch.cuda.empty_cache()

param_grid = {
    "learning_rate": [0.001],
    "batch_size": [16],
    "weight_decay": [0],
    "num_epochs": [200],  
}

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
        data_dir = "/home/ychappetjuan/Alzeimer/datasets"
        csv_path = "/home/ychappetjuan/Alzeimer/list_standardized_tongtong_2017.csv"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Reproductibilité
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = Basic3DCNN(input_channels=input_channels, num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        writer = SummaryWriter(log_dir='./logs') 

        dataset = HippocampusDataset3D(data_dir, csv_path)
        print("Taille totale du dataset :", len(dataset))

        labels = [dataset[i][2] for i in range(len(dataset))]

        run_dir = "/home/ychappetjuan/Alzeimer/models"
        hyperparam_string = f"lr-{learning_rate}_bs-{batch_size}_wd-{weight_decay}_epochs-{num_epochs}"

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
        unique_run_dir = os.path.join(run_dir, f"{current_time}_{hyperparam_string}")
        os.makedirs(unique_run_dir, exist_ok=True)
        checkpoint_dir = os.path.join(unique_run_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Dossier pour le run créé : {unique_run_dir}")
        print(f"Dossier checkpoints créé : {checkpoint_dir}")

        # Lancer 10 entraînements avec des splits différents
        for i in range(10):  # Répéter l'entraînement 10 fois
            print(f"Début de l'entraînement {i + 1}/10")

            # Split des indices pour ce run
            train_indices, val_indices = train_test_split(
                range(len(dataset)),
                test_size=0.2,
                stratify=labels,
                random_state=i  # Utiliser un random_state différent pour chaque itération
            )

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            early_stopping = EarlyStopping(
                patience=5,
                verbose=True,
                path=os.path.join(checkpoint_dir, f'best_model_{i}.pth')
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

            # Enregistrer les résultats pour cette exécution
            with open(results_file, mode='a', newline='') as file:
                writer2 = csv.writer(file)
                writer2.writerow([learning_rate, batch_size, weight_decay, 
                                num_epochs, val_loss, val_accuracy, train_loss, train_accuracy])

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_hyperparams = (learning_rate, batch_size, weight_decay, num_epochs)

            print(f"Meilleurs hyperparamètres jusqu'à présent: {best_hyperparams}")
            print(f"Meilleure précision de validation: {best_val_accuracy:.4f}")

        writer.close()

def main_mixup():
    # Nom du fichier CSV
    results_file = "hyperparameter_results_mixup.csv"

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
        # data_dir = "/home/ychappetjuan/Alzeimer/datasets"
        # csv_path = "/home/ychappetjuan/Alzeimer/list_standardized_tongtong_2017.csv"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Reproductibilité
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model = Basic3DCNN(input_channels=input_channels, num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        writer = SummaryWriter(log_dir='./logs_mixup')

        dataset = torch.load('augmented_dataset.pt')
        print("Taille totale du dataset :", len(dataset))

        labels = [dataset[i][2] for i in range(len(dataset))]
        binned_labels = np.digitize(labels, bins=[0.25, 0.5, 0.75])

        run_dir = "/home/ychappetjuan/Alzeimer/models_mixup"
        hyperparam_string = f"lr-{learning_rate}_bs-{batch_size}_wd-{weight_decay}_epochs-{num_epochs}"

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
        unique_run_dir = os.path.join(run_dir, f"{current_time}_{hyperparam_string}")
        os.makedirs(unique_run_dir, exist_ok=True)
        checkpoint_dir = os.path.join(unique_run_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Dossier pour le run créé : {unique_run_dir}")
        print(f"Dossier checkpoints créé : {checkpoint_dir}")

        # Lancer 10 entraînements avec des splits différents
        for i in range(10):  # Répéter l'entraînement 10 fois
            print(f"Début de l'entraînement {i + 1}/10")

            # Split des indices pour ce run
            train_indices, val_indices = train_test_split(
                range(len(dataset)),
                test_size=0.2,
                stratify=binned_labels,
                random_state=i  # Utiliser un random_state différent pour chaque itération
            )

            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            early_stopping = EarlyStopping(
                patience=7,
                verbose=True,
                path=os.path.join(checkpoint_dir, f'best_model_{i}.pth')
            )

            for epoch in range(num_epochs):
                print(f"Époque {epoch + 1}/{num_epochs}")

                train_loss, train_accuracy = train_one_epoch_mixup(model, train_loader, criterion, optimizer, device)
                print(f"Perte d'entraînement: {train_loss:.4f}, Précision d'entraînement: {train_accuracy:.4f}")

                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

                val_loss, val_accuracy = validate_mixup(model, val_loader, criterion, device)
                print(f"Perte de validation: {val_loss:.4f}, Précision de validation: {val_accuracy:.4f}")

                writer.add_scalar('Loss/Val', val_loss, epoch)
                writer.add_scalar('Accuracy/Val', val_accuracy, epoch)

                early_stopping(val_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            # Enregistrer les résultats pour cette exécution
            with open(results_file, mode='a', newline='') as file:
                writer2 = csv.writer(file)
                writer2.writerow([learning_rate, batch_size, weight_decay, 
                                num_epochs, val_loss, val_accuracy, train_loss, train_accuracy])

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_hyperparams = (learning_rate, batch_size, weight_decay, num_epochs)

            print(f"Meilleurs hyperparamètres jusqu'à présent: {best_hyperparams}")
            print(f"Meilleure précision de validation: {best_val_accuracy:.4f}")

        writer.close()

def plot_gradcam_3d(input_image, cam):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    z, y, x = np.indices(input_image.shape)

    cam_threshold = cam > 0.5 * cam.max()  

    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

    ax.voxels(np.ones_like(input_image), facecolors=plt.cm.gray(input_image), edgecolor='none', alpha=0.5)

    ax.voxels(cam_threshold, facecolors=plt.cm.jet(cam / cam.max()), edgecolor='none', alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("Visualisation 3D de l'image et de la Grad-CAM")
    plt.show()

def test():
    data_dir = "/home/ychappetjuan/Alzeimer/datasets"
    csv_path = "/home/ychappetjuan/Alzeimer/list_standardized_tongtong_2017.csv"
    checkpoint_path = "/home/ychappetjuan/Alzeimer/models/2024-12-06_09-10-27/checkpoints/best_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = TestHippocampusDataset3D(data_dir, csv_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    input_channels = 1
    num_classes = 2
    model = Basic3DCNN(input_channels=input_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    grad_cam = GradCAM3D(model, model.conv4)

    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    cam = grad_cam.generate_cam(images)

    cam = F.interpolate(cam.unsqueeze(1), size=(40, 40, 40), mode='trilinear', align_corners=False)
    cam = cam.squeeze(1)  

    cam = cam.squeeze().cpu().numpy()
    input_image = images[0, 0].cpu().numpy()

    plot_gradcam_3d(input_image.squeeze(), cam.squeeze())

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Précision sur le jeu de test : {accuracy:.4f}")
    print(classification_report(all_labels, all_predictions))

if __name__ == "__main__":
    main_mixup()
