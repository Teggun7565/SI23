import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # TODO: Calcular dimension de salida
        out_dim = self.calc_out_dim(input_dim, kernel_size=3, stride=1, padding=0)
        out_dim = self.calc_out_dim(out_dim, kernel_size=3, stride=1, padding=0)
        out_dim = self.calc_out_dim(out_dim, kernel_size=3, stride=1, padding=0)
        out_dim = self.calc_out_dim(out_dim, kernel_size=3, stride=1, padding=0)

        # TODO: Define las capas de tu red
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*out_dim*out_dim, 256)
        self.fc2 = nn.Linear(256, n_classes)

        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Define la propagacion hacia adelante de tu red
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        proba = F.softmax(logits, dim=1)
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def mi_inferencia(self, x):
        with torch.no_grad():
            self.eval()
            x = x.to(self.device)
            _, proba = self.forward(x.unsqueeze(0))
            pred_class = torch.argmax(proba).item()
        return pred_class

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        models_path = file_path / 'models' / model_name
        assert models_path.exists(), f"El archivo {models_path} no existe"
        # TODO: Carga los pesos de tu red neuronal
        self.load_state_dict(torch.load(models_path))
        self.to(self.device)
