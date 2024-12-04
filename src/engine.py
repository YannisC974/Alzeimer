import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model import Basic3DCNN

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Entraîne le modèle pour une époque.
    
    :param model: Le modèle à entraîner.
    :param dataloader: DataLoader contenant les données d'entraînement.
    :param criterion: Fonction de perte.
    :param optimizer: Optimiseur pour mettre à jour les poids du modèle.
    :param device: Appareil (GPU ou CPU) sur lequel entraîner le modèle.
    :return: La perte moyenne et la précision de cette époque.
    """
    model.train()  # Met le modèle en mode entraînement
    
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    # Boucle sur les lots de données
    for inputs, masks, labels in dataloader:
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        
        optimizer.zero_grad()
    
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """
    Valide le modèle sur l'ensemble de validation.
    
    :param model: Le modèle à valider.
    :param dataloader: DataLoader contenant les données de validation.
    :param criterion: Fonction de perte.
    :param device: Appareil (GPU ou CPU) sur lequel valider le modèle.
    :return: La perte et la précision sur l'ensemble de validation.
    """
    model.eval()  
    
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():  
        for inputs, masks, labels in dataloader:
            inputs, masks, labels = inputs.to(device), masks, labels.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


