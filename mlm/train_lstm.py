import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
from lstm_data import TextDataset, collate_fn
from lstm_models import BiLSTMModel
import wandb

# --- 3. Training Loop ---

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    # Initialize wandb logging
    wandb.init(project="LSTM_cz", config={"epochs": num_epochs, "batch_size": train_loader.batch_size})
    wandb.watch(model, criterion, log="all")

    best_val_loss = float('inf')  # Set initial best loss to infinity
    best_model_wts = None
    save_path = "mlm_teacher.pth"

    model.to(device)
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        bc = 0
        for inputs, targets, indices in train_loader:
            try:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs, indices)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                #scheduler.step()

                train_loss += loss.item()
                bc += 1
                if not bc % 250: print(f"Batch {bc}: ", train_loss/bc)
            except RuntimeError:
                print("Encountered a rare RuntimeError, skipping this batch")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss, correct_top5, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets, indices in val_loader:
                try:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs, indices)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    # Calculate top-5 accuracy
                    _, top5_preds = outputs.topk(5, dim=1)  # Get top-5 predictions
                    targets_expanded = targets.unsqueeze(1).expand_as(top5_preds)  # Expand targets to compare with top-5
                    correct_top5 += (top5_preds == targets_expanded).any(dim=1).sum().item()  # Count correct predictions
                    total += targets.size(0)
                except RuntimeError:
                    print("Encountered a rare RuntimeError, skipping this batch")

        val_loss /= len(val_loader)
        top5_accuracy = correct_top5 / total

        # Log training and validation losses and top-5 accuracy to wandb
        wandb.log({
            "epoch": epoch + 1, 
            "train_loss": train_loss, 
            "val_loss": val_loss,
            "top5_accuracy": top5_accuracy
        })
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # If the validation loss improved, save the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()  # Save the model weights

        # Save the best model weights after training
        if best_model_wts is not None:
            torch.save(best_model_wts, save_path)
            print(f"Best model saved to {save_path}")

    wandb.finish()

from torch.utils.data import random_split


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """
    Creates a learning rate scheduler with a linear warmup and linear decay after warmup.

    Parameters:
    - optimizer: The optimizer to use (e.g., Adam).
    - warmup_steps: Number of steps to perform warmup.
    - total_steps: Total number of training steps (including warmup).
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))  # Linear warmup
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))  # Linear decay after warmup

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --- Main Script ---
def main(file_paths, vocab_size, embedding_dim=512, hidden_dim=256, batch_size=32, num_epochs=30, lr=0.0001):
    # Dataset
    dataset = TextDataset(file_paths)
    
    # Split into training and validation sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, Loss, Optimizer
    model = BiLSTMModel(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None#get_lr_scheduler(optimizer, warmup_steps, total_steps)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model with validation
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device)

if __name__ == "__main__":
    file_paths = [os.path.join("data/text", file) for file in os.listdir("data/text") if file.endswith(".txt")]
    vocab_size = 51865  # Define based on your tokenizer's vocabulary
    main(file_paths, vocab_size)
