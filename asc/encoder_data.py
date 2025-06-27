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
    """Custom Dataset for MNIST that stitches a given number of images together"""

    def __init__(self, base_dataset, num_images=3, length=10000, out_area=256):
        """
        Args:
            base_dataset: A PyTorch dataset (e.g. torchvision.datasets.MNIST)
            num_images: How many images to stitch into one sample
            length: How many stitched samples to generate
            out_area: Approximate pixel area of the resized stitched image
        """
        self.samples = []
        self.num_images = num_images

        # Prepare available indices
        available_indices = list(range(len(base_dataset)))

        for _ in range(length):
            # Ensure enough images left for a sample
            if len(available_indices) < num_images:
                break

            # Sample without replacement
            sampled = random.sample(available_indices, num_images)
            for idx in sampled:
                available_indices.remove(idx)

            # Retrieve and stitch
            imgs, labels = zip(*[base_dataset[i] for i in sampled])
            imgs_tensor = torch.stack(imgs)  # (num_images, 1, 28, 28)
            stitched_img, stitched_lbl = self.stitch_and_resize(
                imgs_tensor, labels, out_area
            )
            self.samples.append((stitched_img, stitched_lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def stitch_and_resize(self, images, labels, out_area):
        """
        Stitches `num_images` grayscale digits horizontally and resizes to approx out_area.
        Returns: stitched image (1, h, w), label tensor (num_images,)
        """
        assert images.shape[0] == self.num_images
        images = images.squeeze(1)  # (num_images, 28, 28)
        stitched = torch.cat(
            list(images), dim=1
        )  # horizontal concat → (28, 28 * num_images)

        stitched = stitched.unsqueeze(0)  # (1, 1, h, w)

        label_tensor = torch.tensor(labels, dtype=torch.long)
        return stitched, label_tensor
