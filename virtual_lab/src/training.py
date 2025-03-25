# src/training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data_processor import DataProcessor
from model import GFactorPredictor
import os


def train_model(csv_path, epochs=100, batch_size=32, learning_rate=0.001):
    # Initialize data processor and load data
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.load_and_process_data(csv_path)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = GFactorPredictor(input_size=9)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}, Test Loss: {test_loss:.4f}')

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            if not os.path.exists('../models'):
                os.makedirs('../models')
            torch.save(model.state_dict(), '../models/best_model.pth')

    print("Training completed!")


if __name__ == "__main__":
    train_model("../data/nanomaterials_g-factor.csv")