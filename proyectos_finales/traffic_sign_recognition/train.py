from dataset import get_dataloaders
# import os
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from network import Network
from plot_losses import PlotLosses
from pathlib import Path
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

# TODO: train your model
def main():
    learning_rate = 1e-4
    num_epochs=100
    # batch_size = 500

    # Set up the dataloaders
    train_loader, val_loader = get_dataloaders()  

    # Set up the device (CPU or GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Instantiate your model
    # model = Network().to(device)
    model = Network()

    # Define your loss function
    criterion = nn.CrossEntropyLoss()

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set up the training loop
    plot_losses = PlotLosses()  

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute the loss
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Compute the accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print the loss and accuracy for this epoch
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save the trained model
        model.save_model('mejor_modelo.pt')

        # Update the plot of losses
        plot_losses.on_epoch_end(train_loss, val_loss)
    plot_losses.on_train_end()

    
if __name__ == "__main__":
    main()
