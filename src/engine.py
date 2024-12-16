import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model import Basic3DCNN
from tqdm import tqdm

def mixup_loss(output, target, lam):
    """
    Fonction de perte pour le mixup. Elle prend en compte les labels mélangés.
    :param output: Sorties du modèle (logits)
    :param target: Labels mélangés
    :param lam: Facteur de mixup utilisé pour pondérer les labels
    :return: La perte mixée
    """
    return lam * F.cross_entropy(output, target[0]) + (1 - lam) * F.cross_entropy(output, target[1])

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
    model.train()  
    
    running_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for batch_idx, (inputs, masks, labels) in enumerate(progress_bar):
        inputs, masks, labels = inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
    
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def train_one_epoch_mixup(model, dataloader, criterion, optimizer, device):
    """
    Entraîne le modèle pour une époque.
    
    :param model: Le modèle à entraîner.
    :param dataloader: DataLoader contenant les données d'entraînement.
    :param criterion: Fonction de perte.
    :param optimizer: Optimiseur pour mettre à jour les poids du modèle.
    :param device: Appareil (GPU ou CPU) sur lequel entraîner le modèle.
    :return: La perte moyenne et la précision de cette époque.
    """
    model.train()  
    
    running_loss = 0.0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for batch_idx, (inputs, masks, labels) in enumerate(progress_bar):
        inputs, masks, labels = inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Si les labels sont mélangés, récupérez aussi le facteur lambda
        if isinstance(labels, tuple):  # Vérifie si les labels sont un tuple, ce qui signifie qu'ils sont mixés
            lam = torch.tensor(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])).to(device)
            loss = mixup_loss(model(inputs), labels, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
    
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
            inputs, masks, labels = inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate_mixup(model, dataloader, criterion, device):
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
            inputs, masks, labels = inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Si les labels sont mélangés, récupérez aussi le facteur lambda
            if isinstance(labels, tuple):  # Vérifie si les labels sont un tuple, ce qui signifie qu'ils sont mixés
                lam = torch.tensor(np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])).to(device)
                loss = mixup_loss(model(inputs), labels, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

