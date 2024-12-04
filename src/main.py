import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from engine import train_one_epoch, validate
from dataset import HippocampusDataset3D

from model import Basic3DCNN 

def main():
    # Paramètres
    input_channels = 1  
    num_classes = 2  
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4
    data_dir = "/Users/yannischappetjuan/Desktop/IA/ALZHEIMER/raw_data"
    csv_path = "/Users/yannischappetjuan/Desktop/IA/ALZHEIMER/raw_data/list_standardized_tongtong_2017.csv"
    device = torch.device('cpu')


    model = Basic3DCNN(input_channels=input_channels, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir='./logs') 

    dataset = HippocampusDataset3D(data_dir, csv_path)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        print(f"Époque {epoch + 1}/{num_epochs}")

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Perte d'entraînement: {train_loss:.4f}, Précision d'entraînement: {train_accuracy:.4f}")

        # Ajouter les métriques d'entraînement à TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        # Valider sur l'ensemble de validation
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Perte de validation: {val_loss:.4f}, Précision de validation: {val_accuracy:.4f}")

        # Ajouter les métriques de validation à TensorBoard
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_accuracy, epoch)


    # Sauvegarder le modèle
    torch.save(model.state_dict(), 'model.pth')
    print("Modèle sauvegardé sous 'model.pth'")

    # Fermer le writer de TensorBoard
    writer.close()

if __name__ == "__main__":
    main()
