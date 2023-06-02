import torch.nn as nn
import torch
from pathlib import Path

file_path = Path(__file__).parent.absolute()

#TODO: Define la red neuronal
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 43)

        self.init_weights()
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def save_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        assert models_path.exists(), f"El archivo {models_path} no existe"
        self.load_state_dict(torch.load(models_path, map_location=self.device))

    def evaluar_inferencia(self, x):
        with torch.no_grad():
            self.eval()
            x = x.to(self.device)
            _, proba = self.forward(x.unsqueeze(0))
            pred_class = torch.argmax(proba).item()
        return pred_class


def main():
    net = Network()
    print(net)
    torch.rand(1, 3, 32, 32)
    print(net(torch.rand(1, 3, 32, 32)))


if __name__ == "__main__":
    main()
