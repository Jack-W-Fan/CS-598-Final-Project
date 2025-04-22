import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_times = []

        
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            
            # Ensure shapes match [batch_size, 1]
            outputs = outputs.view(-1, 1)
            y = y.view(-1, 1)
            
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x).view(-1, 1)
                y = y.view(-1, 1)
                val_loss += criterion(outputs, y).item()
                
                # For accuracy calculation
                predicted = (outputs > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
                
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(correct / total)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, '
              f'Val Acc: {val_accuracies[-1]:.4f}')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    return train_losses, val_losses, val_accuracies

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x).view(-1, 1)
            y = y.view(-1, 1)
            predicted = (outputs > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy