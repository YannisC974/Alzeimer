import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from engine import train_one_epoch, validate
from dataset import HippocampusDataset3D
import matplotlib.pyplot as plt
import numpy as np

from model import Basic3DCNN 

def main():
    # Paramètres
    input_channels = 1
    num_classes = 2
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    data_dir = "/home/ychappetjuan/Alzeimer/datasets"
    csv_path = "/home/ychappetjuan/Alzeimer/list_standardized_tongtong_2017.csv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = Basic3DCNN(input_channels=input_channels, num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir='./logs') 

    dataset = HippocampusDataset3D(data_dir, csv_path)

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

    generator = torch.Generator().manual_seed(42)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Modèle sauvegardé sous '{checkpoint_path}'")

    # Sauvegarder le modèle
    torch.save(model.state_dict(), 'model.pth')
    print("Modèle sauvegardé sous 'model.pth'")

    # Fermer le writer de TensorBoard
    writer.close()

if __name__ == "__main__":
    main()
