import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# --- Config ---
batch_size = 128
epochs = 10
learning_rate = 1e-3
patch_size = 7
embed_dim = 64
num_heads = 4
num_layers = 3
num_classes = 10
data_path = "./data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset & Loader ---
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

train_dataset = datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root=data_path, train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Image Stitching ---
def stitch_and_resize(images, out_size=250):
    """
    images: Tensor of shape (4, 1, 28, 28)
    Returns: Tensor of shape (1, out_size, out_size)
    """
    assert images.shape[0] == 4
    # Squeeze channel for concatenation
    images = images.squeeze(1)  # (4, 28, 28)
    row1 = torch.cat([images[0], images[1]], dim=1)
    row2 = torch.cat([images[2], images[3]], dim=1)
    stitched = torch.cat([row1, row2], dim=0).unsqueeze(0)  # (1, 56, 56)
    # Now resize to (1, out_size, out_size)
    stitched_resized = TF.resize(stitched, [out_size, out_size])
    return stitched_resized

def sample_random_images(dataset, num):
    idxs = random.sample(range(len(dataset)), num)
    imgs = []
    for i in idxs:
        img, _ = dataset[i]  # img: (1, 28, 28)
        imgs.append(img)
    return torch.stack(imgs)

# --- Patch Embedding ---
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=7, embed_dim=64, img_size=28, in_chans=1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, num_patches, C, 7, 7)
        patches = patches.reshape(B, self.num_patches, -1)  # (B, num_patches, patch_size*patch_size*C)
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
            nn.Linear(embed_dim * 4, embed_dim)
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
    def __init__(self, patch_size, embed_dim, num_heads, num_layers, num_classes, img_size=28, in_chans=1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, embed_dim, img_size, in_chans)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.ModuleList([EncoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        x = self.patch_embed(x)           # (B, 16, 64)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 64)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 17, 64)
        x = x + self.pos_embed                        # (B, 17, 64)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0, :]                              # (B, 64)
        out = self.head(cls)                          # (B, num_classes)
        return out


# --- Instantiate Model ---
model = VisualTransformer(
    patch_size=patch_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# --- Training Loop ---
for epoch in range(epochs):
    correct_total, sample_total = 0, 0
    model.train()
    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f} | Accuracy: {epoch_accuracy:.2f}%")

torch.save(model.state_dict(), 'mnist_vit_encoder.pth')

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
    return accuracy

# After training:
test_accuracy = evaluate(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")