import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.nn.utils.rnn import pad_sequence
import random
import math
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
# transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])

train_dataset = datasets.MNIST(root=data_path, train=True, download=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root=data_path, train=False, download=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Image Stitching ---
def stitch_and_resize(images, labels, out_size=img_size):
    """
    images: Tensor of shape (N, 1, 28, 28)
    labels: Tensor of N integers representing the labels of the images
    Returns: Tensor of shape (1, out_size, out_size) and a tensor of labels of shape (N,)
    """
    # Squeeze channel for concatenation
    images = images.squeeze(1)  # (N, 28, 28)
    # Extract the label from each image and append in order of image selection
    labels = torch.tensor(labels)  # (N,)
    N = len(images)
    grid_size = math.ceil(math.sqrt(N))
    pad_needed = grid_size**2 - N
    if pad_needed > 0:
        blank = torch.zeros((28, 28), dtype=images.dtype, device=images.device)
        # Add pad_needed blank images to fill the grid
        images = torch.cat([images, blank.unsqueeze(0).repeat(pad_needed, 1, 1)], dim=0)

    # ---- Perturb each digit before arranging ----
    perturbed_imgs = []
    for img in images:
        pil_img = TF.to_pil_image(img.cpu())

        angle = random.uniform(-45, 45)    # mild rotation, less artifacting
        scale = random.uniform(0.5, 1.2)  # only shrink, never enlarge out of bounds
        max_trans = 5
        translate_x = random.randint(-max_trans, max_trans)
        translate_y = random.randint(-max_trans, max_trans)

        perturbed = TF.affine(
            pil_img,
            angle=angle,
            translate=(0, 0),
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,  # <--- Key change!
            fill=0,
        )
        perturbed = TF.to_tensor(perturbed).squeeze(0)
        # Center crop if needed
        if perturbed.shape[-2:] != (28, 28):
            perturbed = TF.center_crop(perturbed, (28, 28))
        perturbed_imgs.append(perturbed.to(img.device))

    perturbed_imgs = torch.stack(perturbed_imgs)

    rows = []
    for r in range(grid_size):
        row_imgs = perturbed_imgs[r*grid_size:(r+1)*grid_size]  # shape: (cols, 28, 28)
        row_cat = torch.cat(list(row_imgs), dim=1)      # concat horizontally
        rows.append(row_cat)
    
    # Concatenate all rows vertically
    stitched = torch.cat(rows, dim=0).unsqueeze(0)   # vertical, shape: (1, H, W)
    mnist_mean, mnist_std = 0.1307, 0.3081
    # Now resize to (1, out_size, out_size)
    stitched_resized = TF.resize(stitched, [out_size, out_size])
    stitched_resized = (stitched_resized - mnist_mean) / mnist_std
    return stitched_resized, labels

# --- Custom Dataset ---
class CustomMNISTDataset(torch.utils.data.Dataset):
    """ Custom Dataset for MNIST that stitches a given number of images together """
    def __init__(self, mnist_dataset, length=60000):
        self.mnist_dataset = mnist_dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get N random images and their labels
        num = torch.randint(1, 11, (1,)).item()  # Randomly choose between 1 and 10
        idxs = torch.randint(0, len(self.mnist_dataset), (num,)) # Random indices
        # Get images and labels from the dataset
        imgs, labels = zip(*(self.mnist_dataset[i.item()] for i in idxs))
        labels = list(labels) + [11]
        images = torch.stack(imgs)
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
    def __init__(self, patch_size, embed_dim, num_heads, num_layers, num_classes, img_size=img_size, in_chans=1, seq_len=11):
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
        self.vocab_size = num_classes + 3   # +1 for start token
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
        curr_seq_len = y.shape[1]
        pos_encod_dec = self.pos_encod_dec[:, :curr_seq_len, :].expand(B, curr_seq_len, -1)
        y = y + pos_encod_dec                           # (B, seq_len, embed_dim)
        mask = torch.triu(torch.ones((curr_seq_len, curr_seq_len), device=x.device), diagonal=1).bool()
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
            mask = y_tar != 12  # Ignore padding index
            correct += (preds[mask] == y_tar[mask]).sum().item()
            total += mask.sum().item()

    accuracy = 100 * correct / total
    return accuracy

def collate_fn(batch):
    x_seqs, y_seqs = zip(*batch)
    y_lens = [y.shape[0] for y in y_seqs]
    x_batch = torch.stack(x_seqs)
    y_padded = pad_sequence(y_seqs, batch_first=True, padding_value=12)
    return x_batch, y_padded, y_lens

# --- Build Custom Dataset and DataLoader ---
train_dataset_stitch = CustomMNISTDataset(train_dataset, length=300000)
train_loader_stitch = DataLoader(train_dataset_stitch, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

test_dataset_stitch = CustomMNISTDataset(test_dataset, length=50000)
test_loader_stitch = DataLoader(test_dataset_stitch, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


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
loss_fn = nn.CrossEntropyLoss(ignore_index=12)  # 12 is the padding index for y_padded

# --- Training Loop ---
for epoch in range(epochs):
    correct_total, sample_total = 0, 0
    model.train()
    for x_batch, y_batch, y_lens in tqdm(train_loader_stitch, desc=f"Epoch {epoch+1}/{epochs}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        B = x_batch.size(0)
        start_tokens = torch.full((B, 1), 10, dtype=y_batch.dtype, device=y_batch.device)
        y_input = torch.cat([start_tokens, y_batch[:, :-1]], dim=1)
        y_target = y_batch.clone() # (B, seq_len)
        logits = model(x_batch, y_input) # (B, seq_len, vocab_size)

        # Flatten loss & y_target to avoid loss not averaging across all tokens in all batches
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size) # (B * seq_len, vocab_size)
        y_target = y_target.reshape(-1) # (B * seq_len)
        loss = loss_fn(logits, y_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        mask = y_target != 12
        correct_total += (preds[mask] == y_target[mask]).sum().item()
        sample_total += mask.sum().item()

    epoch_accuracy = (correct_total / sample_total) * 100
    print(f"Epoch {epoch+1}: Loss {loss.item():.4f} | Accuracy: {epoch_accuracy:.2f}%")

torch.save(model.state_dict(), 'mnist_vit_multi_stitch_pert.pth')

# After training:
test_accuracy = evaluate(model, test_loader_stitch)
print(f"Test Accuracy: {test_accuracy:.2f}%")