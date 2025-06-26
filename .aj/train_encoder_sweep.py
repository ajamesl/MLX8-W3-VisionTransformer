import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

# --- Config ---
batch_size = 128
epochs = 10
learning_rate = 1e-3
patch_size = 7
embed_dim = 64
num_heads = 4
num_layers = 3
num_classes = 10
img_size = 28
data_path = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset & Loader ---
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


# --- Patch Embedding ---
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=patch_size, embed_dim=embed_dim, img_size=img_size, in_chans=1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, num_patches, C, 7, 7)
        patches = patches.reshape(
            B, self.num_patches, -1
        )  # (B, num_patches, patch_size*patch_size*C)
        return self.proj(patches)  # (B, num_patches, embed_dim)

    # --- Encoder Block ---


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        x_res1 = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x)
        x = x + x_res1

        x_res2 = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = x + x_res2
        return x


# --- Visual Transformer ---
class VisualTransformer(nn.Module):
    def __init__(
        self,
        patch_size,
        embed_dim,
        num_heads,
        num_layers,
        num_classes,
        img_size=28,
        in_chans=1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, embed_dim, img_size, in_chans)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.ModuleList(
            [EncoderBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, 16, 64)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 64)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 17, 64)
        x = x + self.pos_embed  # (B, 17, 64)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0, :]  # (B, 64)
        out = self.head(cls)  # (B, num_classes)
        return out

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.005},
        'batch_size': {'values': [64, 128]},
        'num_heads': {'values': [2, 4, 8]},
        'num_layers': {'values': [2, 4, 6]},
        'embed_dim': {'values': [64, 128]},
        'epochs': {'value': 10},
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="vit-mnist", entity="mlx-aj")

def train():
    run = wandb.init()
    config = wandb.config

    learning_rate = config.learning_rate
    batch_size = config.batch_size
    num_heads = config.num_heads
    num_layers = config.num_layers
    embed_dim = config.embed_dim
    num_epochs = config.epochs

    # --- Instantiate Model ---
    model = VisualTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_dataset = datasets.MNIST(
        root=data_path, train=True, download=False, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Training Loop ---
    for epoch in range(num_epochs):
        correct_total, sample_total, epoch_loss = 0, 0, 0
        model.train()
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct_total += (preds == y_batch).sum().item()
            sample_total += len(y_batch)
        epoch_accuracy = (correct_total / sample_total) * 100
        print(
            f"Epoch {epoch + 1}: Loss {loss.item():.4f} | Accuracy: {epoch_accuracy:.2f}%"
        )
        
        epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)

        wandb.log({
            "epoch": epoch + 1,
            "loss": avg_epoch_loss,
            "epoch accuracy": epoch_accuracy,
        })

    run.finish()

# This launches the sweep agent, and will run as many runs as you want
if __name__ == "__main__":
    wandb.agent(sweep_id, train, count=20)   # or count=None for infinite runs