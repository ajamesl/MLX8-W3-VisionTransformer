import torch
import torch.nn as nn
import wandb
from config import get_hyperparams
from encoder_data import CustomMNISTDataset
from encoder_model import Transformer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.utils as utils




hp = get_hyperparams()

batch_size = hp["batch_size"]
epochs = hp["epochs"]
lr = hp["learning_rate"]
patch_size = hp["patch_size"]
embed_dim = hp["embed_dim"]
num_heads = hp["num_heads"]
num_layers = hp["num_layers"]
num_classes = hp["num_classes"]
img_size = hp["img_size"]
data_path = hp["data_path"]
seq_len = hp["seq_len"]
num_patches = hp["num_patches"]

# --- Dataset & Loader ---
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_dataset = datasets.MNIST(
    root=data_path, train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root=data_path, train=False, download=True, transform=transform
)

train_dataset_stitch = CustomMNISTDataset(train_dataset, length=60_000)
test_dataset_stitch = CustomMNISTDataset(test_dataset, length=4_000)

#print(len(train_dataset_stitch))
#print(len(test_dataset))

train_loader = DataLoader(train_dataset_stitch, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset_stitch, batch_size=batch_size, shuffle=False)


img, label = train_dataset_stitch[0]
print("Image shape is:", img.shape)  # should be (1, h, w)
print("Label shape is:", label)  # should be tensor of shape (3,)

img, label = train_dataset_stitch[0]
plt.imshow(img.squeeze(0), cmap="gray")
plt.title(f"Labels: {label.tolist()}")
plt.axis("off")
plt.savefig("sample.png")
plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(
    project="mlx6-week-03-VIZZYBOP",
    config={
        "num_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "embedding_dim": embed_dim,
    },
)

model = Transformer(
    embed_dim=embed_dim,
    patch_size=patch_size,
    img_size=img_size,
    num_classes=num_classes,
    num_layers=num_layers,
    num_heads=num_heads,
    seq_len=seq_len
).to(device)

optimiser = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


# training loop

global_step = 0

for epoch in range(epochs):
    model.train()
    correct_total, sample_total = 0, 0
    asc_correct, asc_total = 0, 0
    desc_correct, desc_total = 0, 0
    running_loss = 0.0
    grad_norm_total = 0.0
    
    first_image_logged = False  # Flag to print one sample only

    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        #print(f"x_batch.shape: {x_batch.shape}")
        logits = model(x_batch, y_batch)
        # Reshape for CrossEntropy: (B*3, 10) vs (B*3,)
        #print("logits.shape:", logits.shape)
        #print("logits.shape before view:", logits.shape)
        vocab_size = logits.shape[-1]  # Should be 13 (12 classes + start token)
        loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))

        optimiser.zero_grad()
        loss.backward()
        
        # Compute total gradient norm (L2 norm across all parameters)
        total_norm = torch.norm(
            torch.stack([
                param.grad.norm(2) for param in model.parameters() if param.grad is not None
            ])
        ).item()
        grad_norm_total += total_norm

        
        optimiser.step()
        
        preds = logits.argmax(dim=2)  # (B, 4)

        # Full sequence match
        correct_sequences = (preds == y_batch).all(dim=1)

        correct_total += correct_sequences.sum().item()
        sample_total += x_batch.size(0)
        running_loss += loss.item() * x_batch.size(0)

        # Asc / Desc accuracy
        labels_type = y_batch[:, -1]  # last value in sequence: 10 or 11
        asc_mask = labels_type == 10
        desc_mask = labels_type == 11

        asc_correct += ((preds == y_batch).all(dim=1) & asc_mask).sum().item()
        asc_total += asc_mask.sum().item()

        desc_correct += ((preds == y_batch).all(dim=1) & desc_mask).sum().item()
        desc_total += desc_mask.sum().item()

        wandb.log({"train_loss": loss.item()}, step=global_step)

        # Print one sample per epoch
        if not first_image_logged:
            sample_img = x_batch[0].cpu()
            sample_true = y_batch[0].cpu().tolist()
            sample_pred = preds[0].cpu().tolist()

            print(f"\n[Sample image]")
            plt.imshow(sample_img.squeeze(0), cmap="gray")
            plt.title(f"Label: {sample_true} | Prediction: {sample_pred}")
            plt.axis("off")
            plt.savefig(f"epoch_{epoch+1}_sample.png")
            plt.close()
            print(f"Label:     {sample_true}")
            print(f"Prediction:{sample_pred}")
            first_image_logged = True

        global_step += 1

    epoch_accuracy = (correct_total / sample_total) * 100
    avg_loss = running_loss / sample_total
    avg_grad_norm = grad_norm_total / len(train_loader)

    asc_accuracy = 100 * asc_correct / max(1, asc_total)
    desc_accuracy = 100 * desc_correct / max(1, desc_total)

    print(f"\nEpoch {epoch + 1}:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {epoch_accuracy:.2f}%")
    print(f"  Ascending Accuracy: {asc_accuracy:.2f}%")
    print(f"  Descending Accuracy: {desc_accuracy:.2f}%")
    print(f"  Avg Gradient Norm: {avg_grad_norm:.4f}")

    wandb.log({
        "epoch_accuracy": epoch_accuracy,
        "epoch_avg_loss": avg_loss,
        "epoch_grad_norm": avg_grad_norm,
        "asc_accuracy": asc_accuracy,
        "desc_accuracy": desc_accuracy
    }, step=global_step)
        


torch.save(model.state_dict(), "encoder_model1.pth")


def evaluate(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x,y)
            preds = logits.argmax(dim=2)  # (B, 3), same as train
            correct += (preds == y).all(dim=1).sum().item()
            total += x.size(0)
    accuracy = 100 * correct / total
    wandb.log({"test_seq_accuracy": accuracy})
    return accuracy


# After training:
test_accuracy = evaluate(model, test_loader)
wandb.log({"test_accuracy": test_accuracy})
print(f"Test Accuracy: {test_accuracy:.2f}%")

wandb.finish()
