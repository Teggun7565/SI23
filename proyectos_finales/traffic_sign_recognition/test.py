import matplotlib.pyplot as plt
import cv2
from network import Network
import torch
from pathlib import Path
import pandas as pd
from dataset import get_transform


def load_img(path):
    if isinstance(path, str):
        path = Path(path)
    assert path.is_file(), f"El archivo {path} no existe"
    img = cv2.imread(path.as_posix())
    transform = get_transform()
    return transform(img)


def visualize_test(test_images, pred, labels):
    '''
        Visualiza los resultados de la inferencia
        args:
        - test_images (torch.Tensor): imagenes de test
        - pred (torch.Tensor): predicciones (números)
    '''
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.axis("off")
        plt.imshow(test_images[i].permute(1, 2, 0))
        plt.title(f"Pred: {pred[i].item()}, Label: {labels[i].item()}")
    plt.show()


if __name__ == "__main__":
    file_path = Path(__file__).parent.absolute()

    # TODO: load your model
    model = Network()
    model.load_model("mejor_modelo.pt")

    test_data = file_path / "data/test_data/test_data/"
    test_labels = pd.read_csv(file_path / "data/test_labels.csv")

    
    image_files = test_labels["image"]

    labels = torch.from_numpy(test_labels["label"].to_numpy())
    test_images = []
    for image_file in image_files:
        img = load_img(test_data / image_file)
        test_images.append(img)
    test_images = torch.stack(test_images)

    # TODO: Evaluate your model with all the files in test 
    with torch.no_grad():
        outputs = model(test_images)
        predictions = torch.argmax(outputs, dim=1)

    # and visualize some of the results using visualize_test
    visualize_test(test_images, predictions, labels)

