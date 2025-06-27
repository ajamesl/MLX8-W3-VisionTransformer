import torch
import torch.nn as nn
import wandb
from encoder_data import (
    load_normalised_dataset,
    patch_to_vector,
    patchlabel,
)
from encoder_model import Transformer
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

# hyperparam
patch_size = 7
num_patches = 16
embed_dim = 64  #
patch_dim = 49  # patch_size * patch_size
batch_size = 512
num_epochs = 10
learning_rate = 0.001
data_path = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="mlx6-week-03-viz",
    config={
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "embedding_dim": embed_dim,
    },
)

## dataset &

train_dataset, test_dataset = load_normalised_dataset(
    datasets.MNIST, data_dir=data_path
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

all_images = [(img[0], label) for img, label in train_dataset]
all_patches, all_labels = patchlabel(all_images)
dataset_vectors = patch_to_vector(all_patches)

model = Transformer(
    patch_dim=patch_dim,
    embed_dim=embed_dim,
    num_patches=num_patches,
    # num_heads=4,
    num_classes=10,
    num_layers=6,
).to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# training loop

global_step = 0

for epoch in range(num_epochs):
    correct_total, sample_total = 0, 0
    running_loss = 0.0
    model.train()
    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        predictions = logits.argmax(dim=1)
        correct_total += (predictions == y_batch).sum().item()

        running_loss += loss.item() * x_batch.size(0)
        sample_total += len(y_batch)
        # Log per batch with global_step
        wandb.log({"train_loss": loss.item()}, step=global_step)
        global_step += 1

    epoch_accuracy = (correct_total / sample_total) * 100
    avg_loss = running_loss / sample_total

wandb.log(
    {"epoch_accuracy": epoch_accuracy, "epoch_avg_loss": avg_loss}, step=global_step
)
print(f"Epoch {epoch + 1}: Loss {loss.item():.4f} | Accuracy: {epoch_accuracy:.2f}%")

torch.save(model.state_dict(), "encoder_model1.pth")


def evaluate(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    accuracy = 100 * correct / total
    wandb.log({"train_loss": loss.item(), "accuracy": accuracy})
    return accuracy


# After training:
test_accuracy = evaluate(model, test_loader)
wandb.log({"test_accuracy": test_accuracy})
print(f"Test Accuracy: {test_accuracy:.2f}%")

wandb.finish()
