from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

# load and normalise the MNIST dataset


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


def split_image(image: torch.Tensor, patch_size: int = 7) -> torch.Tensor:
    """
    Split a (1, 28, 28) image into non-overlapping patches of shape (patch_size x patch_size),
    returning shape (num_patches, patch_size, patch_size)
    """
    if image.dim() == 4:
        image = image.squeeze(0)  # e.g. (1, 1, 28, 28) → (1, 28, 28)

    if image.dim() == 3:
        c, h, w = image.shape
    elif image.dim() == 2:
        c = 1
        h, w = image.shape
        image = image.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    assert h == 28 and w == 28, f"Expected 28x28 image, got {h}x{w}"

    # Use unfold to create (num_patches_h, num_patches_w, patch_size, patch_size)
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Result: (1, 4, 4, 7, 7)
    patches = patches.contiguous().view(-1, patch_size, patch_size)
    return patches  # (16, 7, 7)


def patchlabel(
    all_images: list[tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_patches = [
        split_image(img, 7).squeeze(0)
        if split_image(img, 7).dim() == 4
        else split_image(img, 7)
        for img, _ in all_images
    ]
    all_patches = torch.stack(all_patches)
    all_labels = torch.tensor([label for _, label in all_images])
    return all_patches, all_labels


def patch_to_vector(patches: torch.Tensor) -> torch.Tensor:
    N, num_patches, H, W = patches.shape
    assert H == 7 and W == 7
    return patches.view(N, num_patches, -1)  # → (N, num_patches, 49)


def embed_dataset(dataset: torch.Tensor, patch_embed: nn.Module) -> torch.Tensor:
    return patch_embed(dataset)  # (B, N, D)


def add_cls_and_position_embedding(
    embedded: torch.Tensor, cls_token: torch.Tensor, position_embed: torch.Tensor
) -> torch.Tensor:
    B, N, D = embedded.shape
    cls_tokens = cls_token.expand(B, -1, -1)
    return torch.cat((cls_tokens, embedded), dim=1) + position_embed


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
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, num_patches, C, 7, 7)
        patches = patches.reshape(
            B, self.num_patches, -1
        )  # (B, num_patches, patch_size*patch_size*C)
        return self.proj(patches)  # (B, num_patches, embed_dim)
