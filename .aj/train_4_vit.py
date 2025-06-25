import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm

# --- Config ---
batch_size = 128
epochs = 10
learning_rate = 1e-3
patch_size = 16
embed_dim = 64
num_heads = 4
num_layers = 3
num_classes = 10
img_size = 256
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
def stitch_and_resize(images, labels, out_size=img_size):
    """
    images: Tensor of shape (4, 1, 28, 28)
    labels: Tensor of 4 integers representing the labels of the images
    Returns: Tensor of shape (1, out_size, out_size) and a tensor of labels of shape (4,)
    """
    assert images.shape[0] == 4
    # Squeeze channel for concatenation
    images = images.squeeze(1)  # (4, 28, 28)
    # Extract the label from each image and append in order of image selection
    labels = torch.tensor(labels)  # (4,)
    row1 = torch.cat([images[0], images[1]], dim=1)
    row2 = torch.cat([images[2], images[3]], dim=1)
    stitched = torch.cat([row1, row2], dim=0).unsqueeze(0)  # (1, 56, 56)
    # Now resize to (1, out_size, out_size)
    stitched_resized = TF.resize(stitched, [out_size, out_size])
    return stitched_resized, labels

def sample_random_images(dataset, num=4):
    """ Function to sample (4) random images from the dataset 
    Returns: Tensor of shape (num, 1, 56, 56) """
    idxs = random.sample(range(len(dataset)), num)
    imgs, labels = [], []
    for i in idxs:
        img, label = dataset[i]  # img: (1, 56, 56)
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs), labels

# --- Custom Dataset ---
class CustomMNISTDataset(torch.utils.data.Dataset):
    """ Custom Dataset for MNIST that stitches a given number of images together """
    def __init__(self, mnist_dataset, length=60000, num_images=4):
        self.mnist_dataset = mnist_dataset
        self.length = length
        self.num_images = num_images

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get 4 random images and their labels
        images, labels = sample_random_images(self.mnist_dataset, num=self.num_images)
        stitched_image, stitched_label = stitch_and_resize(images, labels)
        return stitched_image, stitched_label

# --- Patch Embedding ---
class PatchEmbed(nn.Module):
    """ Patch Embedding Layer for Vision Transformer
    Args:"""
    def __init__(self, patch_size=patch_size, embed_dim=embed_dim, img_size=img_size, in_chans=1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x):
        # x: (B, 1, 256, 256)
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches: (B, C, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        # patches: (B, C, num_patches, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, num_patches, C, 16, 16)
        patches = patches.reshape(B, self.num_patches, -1)  # (B, num_patches, patch_size*patch_size*C) / (B, 256, 256)
        return self.proj(patches)  # (B, 256, embed_dim)
    
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

# --- Decoder Block ---
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.masked_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, enc_out, mask):
        x_res1 = x
        x = self.ln1(x)
        x, _ = self.masked_attn(x, x, x, attn_mask=mask)
        x = x + x_res1

        x_res2 = x
        x, _ = self.cross_attn(x, enc_out, enc_out)

        x = self.ln2(x)
        x = x + x_res2

        x_res3 = x
        x = self.mlp(x)
        x = x + x_res3
        return x
    
# --- Visual Transformer ---
class VisualTransformer(nn.Module):
    def __init__(self, patch_size, embed_dim, num_heads, num_layers, num_classes, img_size=img_size, in_chans=1, seq_len=4):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, embed_dim, img_size, in_chans)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_encod_enc = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_encod_enc, std=0.02)

        self.encoder = nn.ModuleList([EncoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.seq_len = seq_len
        self.vocab_size = num_classes + 1   # +1 for start token
        self.tok_embed = nn.Embedding(self.vocab_size, embed_dim)
        self.pos_encod_dec = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_encod_dec, std=0.02)

        self.decoder = nn.ModuleList([DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.linear = nn.Linear(embed_dim, self.vocab_size)


    def forward(self, x, y):
        # x: (B, 1, 256, 256)
        B = x.shape[0]
        x = self.patch_embed(x)                        # (B, 256, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 257, embed_dim)
        x = x + self.pos_encod_enc                     # (B, 257, embed_dim)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)                               # (B, 257, embed_dim)

        # y: (B, seq_len)
        y = self.tok_embed(y)                                 # (B, seq_len, embed_dim)
        pos_encod_dec = self.pos_encod_dec.expand(B, -1, -1)  # (B, seq_len, embed_dim)
        y = y + pos_encod_dec                           # (B, seq_len, embed_dim)
        mask = torch.tril(torch.ones((self.seq_len, self.seq_len), device=x.device)).bool()
        for block in self.decoder:
            y = block(y, x, mask=mask)      # (B, seq_len, embed_dim)
        out = self.linear(y)                # (B, seq_len, vocab_size)
        return out

def evaluate(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            start_toks = torch.full((B, 1), 10, dtype=y.dtype, device=y.device)
            y_inp = torch.cat([start_toks, y[:, :-1]], dim=1)
            y_tar = y.clone() # (B, seq_len)
            logits = model(x, y_inp)

            voc_size = logits.size(-1)
            logits = logits.reshape(-1, voc_size) # (B * seq_len, vocab_size)
            y_tar = y_tar.reshape(-1) # (B * seq_len)
            preds = logits.argmax(dim=1)
            correct += (preds == y_tar).sum().item()
            total += y_tar.numel()

    accuracy = 100 * correct / total
    return accuracy


# --- Build Custom Dataset and DataLoader ---
train_dataset_stitch = CustomMNISTDataset(train_dataset, num_images=4, length=120000)
train_loader_stitch = DataLoader(train_dataset_stitch, batch_size=batch_size, shuffle=True)

test_dataset_stitch = CustomMNISTDataset(test_dataset, num_images=4, length=20000)
test_loader_stitch = DataLoader(test_dataset_stitch, batch_size=batch_size, shuffle=False)


# --- Instantiate Model ---
model = VisualTransformer(
    patch_size=patch_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    num_classes=num_classes,
    img_size=img_size,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# --- Training Loop ---
for epoch in range(epochs):
    correct_total, sample_total = 0, 0
    model.train()
    for x_batch, y_batch in tqdm(train_loader_stitch, desc=f"Epoch {epoch+1}/{epochs}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        B = x_batch.size(0)
        start_tokens = torch.full((B, 1), 10, dtype=y_batch.dtype, device=y_batch.device)
        y_input = torch.cat([start_tokens, y_batch[:, :-1]], dim=1)
        y_target = y_batch.clone() # (B, seq_len)
        logits = model(x_batch, y_input) # (B, seq_len, vocab_size)

        print("Logits stats:", logits.min().item(), logits.max().item(), logits.mean().item())
        print("y_target stats:", y_target.min().item(), y_target.max().item())
        print("Any NaN logits?", torch.isnan(logits).any().item())
        print("Any NaN targets?", torch.isnan(y_target.float()).any().item())

        # Flatten loss & y_target to avoid loss not averaging across all tokens in all batches
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size) # (B * seq_len, vocab_size)
        y_target = y_target.reshape(-1) # (B * seq_len)

        print("Logits stats flat:", logits.min().item(), logits.max().item(), logits.mean().item())
        print("y_target stats flat:", y_target.min().item(), y_target.max().item())
        print("Any NaN logits flat?", torch.isnan(logits).any().item())
        print("Any NaN targets flat?", torch.isnan(y_target.float()).any().item())
        loss = loss_fn(logits, y_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        correct_total += (preds == y_target).sum().item()
        sample_total += y_target.numel()

    epoch_accuracy = (correct_total / sample_total) * 100
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f} | Accuracy: {epoch_accuracy:.2f}%")

torch.save(model.state_dict(), 'mnist_vit_4_enc_dec.pth')

# After training:
test_accuracy = evaluate(model, test_loader_stitch)
print(f"Test Accuracy: {test_accuracy:.2f}%")