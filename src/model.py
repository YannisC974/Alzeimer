import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic3DCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        """
        Un CNN 3D basique pour classification des hippocampes.
        :param input_channels: Nombre de canaux en entrée (1 pour un hippocampe).
        :param num_classes: Nombre de classes de sortie (2 pour classification binaire AD/non-AD).
        """
        super(Basic3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(self._get_fc_input_size(), 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout pour éviter le surapprentissage
        self.dropout = nn.Dropout(0.3)
    
    def _get_fc_input_size(self):
        """
        Calcule dynamiquement la taille d'entrée des couches fully connected.
        """
        x = torch.zeros(1, 1, 40, 40, 40)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return torch.flatten(x, start_dim=1).size(1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x