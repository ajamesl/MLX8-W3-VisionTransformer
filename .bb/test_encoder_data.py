import pytest
import torch
from encoder_train import (
    train_dataset
)
from encoder_data import (
    add_cls_and_position_embedding,
    load_normalised_dataset,
    patch_to_vector,
    patchlabel,
    split_image,
)
from encoder_model import (
    Encoder,
    PatchProjector,
    SelfAttentionMulti,
    SelfAttentionSingle,
    Transformer,
)
from torchvision import datasets


## download data to use in tests ##
@pytest.fixture(scope="module")
def mnist_dataset():
    train_ds, test_ds = load_normalised_dataset(datasets.MNIST, data_dir="./data")
    return train_ds, test_ds


def test_load_normalised_dataset(mnist_dataset):
    train_ds, _ = mnist_dataset
    img, label = train_ds[0]
    assert img.shape == torch.Size([1, 28, 28])
    assert isinstance(label, int)
    assert isinstance(train_ds.data, torch.Tensor)
    assert train_ds.data.shape == torch.Size([60000, 28, 28])


def test_split_image(mnist_dataset):
    train_ds, _ = mnist_dataset
    img, label = train_ds[0]  # shape: (1, 28, 28)

    patches = split_image(img, patch_size=7)  # Expected: (16, 7, 7)

    assert isinstance(patches, torch.Tensor)
    assert patches.shape == torch.Size([16, 7, 7])
    assert isinstance(label, int)

    # Optional: check patch content and shape consistency
    for patch in patches:
        assert patch.shape == (7, 7)


def test_patchlabel(mnist_dataset):
    all_images = [(img[0], label) for img, label in train_dataset]
    all_patches, all_labels = patchlabel(all_images)
    assert isinstance(all_patches, torch.Tensor)
    assert isinstance(all_labels, torch.Tensor)
    assert all_patches.shape == torch.Size([60000, 16, 7, 7])
    assert all_labels.shape == torch.Size([60000])


def test_patch_to_vector(mnist_dataset):
    all_images = [(img[0], label) for img, label in train_dataset]
    all_patches, all_labels = patchlabel(all_images)
    dataset_vectors = patch_to_vector(all_patches)
    assert dataset_vectors.shape == torch.Size([60000, 16, 49])
    assert isinstance(dataset_vectors, torch.Tensor)


# Test encoder_training.py


def test_add_cls_and_position_embedding():
    embedded = torch.randn(1, 16, 64)
    cls_token = torch.randn(1, 1, 64)
    pos_embed = torch.randn(1, 17, 64)
    output = add_cls_and_position_embedding(embedded, cls_token, pos_embed)
    assert output.shape == (1, 17, 64)
    # Subtract positional embedding to get raw concatenated tensor
    raw_output = output - pos_embed
    assert torch.allclose(raw_output[0, 0], cls_token.squeeze(0).squeeze(0), atol=1e-5)


# Testing encoder_model.py


def test_patchprojector(mnist_dataset, patch_dim=49, emb_dim=64):
    all_images = [(img[0], label) for img, label in train_dataset]
    all_patches, all_labels = patchlabel(all_images)
    dataset_vectors = patch_to_vector(all_patches)
    example_patches = dataset_vectors[0].float()
    patch_proj = PatchProjector(patch_dim, emb_dim)
    example_output = patch_proj(example_patches)
    assert example_output.shape == torch.Size([16, 64])


def test_selfatten():
    attn = SelfAttentionSingle(embed_dim=64)
    x = torch.randn(2, 16, 64)  # (batch, tokens, embed dim)
    out = attn(x)
    assert (out.shape) == (2, 16, 64)


def test_selfattenmulti():
    attn = SelfAttentionMulti(embed_dim=64, num_heads=4)
    x = torch.randn(2, 16, 64)  # (batch, tokens, embed dim)
    out = attn(x)
    assert (out.shape) == (2, 16, 64)


def test_encoder():
    batch_size = 2
    num_tokens = 9  # e.g., 8 patches + 1 CLS token
    embed_dim = 64

    model = Encoder(embed_dim)
    x = torch.randn(batch_size, num_tokens, embed_dim)

    with torch.no_grad():
        out = model(x)

    # Check shape is preserved
    assert out.shape == x.shape, "Output shape must match input shape."

    # Check that the output is not exactly equal to the input (i.e., model actually changes it)
    assert not torch.allclose(out, x), (
        "Output should differ from input (model should do something)."
    )

    # Optionally, check the output is finite
    assert torch.isfinite(out).all(), "Output contains NaNs or Infs!"


def test_transformer_forward_shape():
    model = Transformer(
        patch_dim=49,
        embed_dim=64,
        num_patches=16,
        # num_heads=4,
        num_layers=6,
    )
    dummy_input = torch.randn(2, 16, 49)  # B=2, N=16, patch_dim=49
    output = model(dummy_input)
    assert output.shape == (2, 10)
