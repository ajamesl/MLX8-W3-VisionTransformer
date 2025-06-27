import random

import torch
import torch.nn as nn
from config import get_hyperparams

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
num_patches = hp["num_patches"]

# load and normalise the MNIST dataset
"""
def load_normalised_dataset(
    dataset_cls: Type[Dataset],
    data_dir: str = "./data",
    train_transform: Optional[transforms.Compose] = None,
    test_transform: Optional[transforms.Compose] = None,
    download: bool = True,
) -> Tuple[Dataset, Dataset]:
    if train_transform is None:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
            ]
        )

    if test_transform is None:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
            ]
        )

    train_dataset = dataset_cls(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    test_dataset = dataset_cls(
        root=data_dir, train=False, download=download, transform=test_transform
    )

    return train_dataset, test_dataset
"""


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim, img_size, in_chans):
        super().__init__()
        img_height, img_width = img_size
        num_patches = (img_height // patch_size) * (img_width // patch_size)
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


class CustomMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, length=20000):
        self.base_dataset = base_dataset
        self.length = length
        self.samples = []
        
        # Precompute digit indices for efficient sampling
        self.digit_indices = {d: [] for d in range(10)}
        for idx, (_, label) in enumerate(base_dataset):
            self.digit_indices[label].append(idx)
        
        # Generate equal number of asc/desc sequences
        for _ in range(length // 2):
            # Generate ascending sequence
            self.samples.append(self._generate_sequence(ascending=True))
            # Generate descending sequence
            self.samples.append(self._generate_sequence(ascending=False))
    
    def _generate_sequence(self, ascending):
        if ascending:
            # Generate strictly increasing sequence
            digits = sorted(random.sample(range(0, 10), 3))
            seq_label = 10
        else:
            # Generate strictly decreasing sequence
            digits = sorted(random.sample(range(0, 10), 3), reverse=True)
            seq_label = 11
            
        d1, d2, d3 = digits
        
        # Get random images for selected digits
        idx1 = random.choice(self.digit_indices[d1])
        idx2 = random.choice(self.digit_indices[d2])
        idx3 = random.choice(self.digit_indices[d3])
        
        img1, _ = self.base_dataset[idx1]
        img2, _ = self.base_dataset[idx2]
        img3, _ = self.base_dataset[idx3]
        
        # Stitch images horizontally
        stitched = torch.cat([img1, img2, img3], dim=2)  # (1, 28, 84)
        labels = torch.tensor([d1, d2, d3, seq_label], dtype=torch.long)
        
        return stitched, labels

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
