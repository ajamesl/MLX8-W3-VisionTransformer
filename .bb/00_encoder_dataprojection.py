import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
%matplotlib inline
import numpy


# load and normalise the MNIST dataset

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# check the shape of the first image
img, label = train_dataset[0]
print(img.shape)    # should be torch.Size([1, 28, 28])
print(type(label))
# check the shape of the dataset
print (train_dataset.data.shape)
print (type(train_dataset.data))

# define function to split an individualimage
def split_image(images, height, width):
    
    """
    Takes Tensor of Grayscale pixel maps of 28 x 28 which have been normalized
    Split original dataset images into patches
    iterates through the images and split them into patches of size (height, width)
    Returns a tensor of patches of shape (N, 1, height, width)
    
    Arguments:
    images -- tensor of shape (N, 1, 28, 28) where N is the number of images
    height - the required height of the patches
    width - the required width of the patches
    
    """    

    patch_height = height
    patch_width = width

    patches = images.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
    patches = patches.contiguous().view(1,-1, patch_height, patch_width)
    return patches.squeeze(0)

# check split_image function

dataset_test_patch = train_dataset[0]
img, label = dataset_test_patch
dataset_test_patch = split_image(img[0], 7, 7)
print(dataset_test_patch.shape)
print(type(dataset_test_patch))
print(label)


# iterate through the dataset and split the images into patches
# create a list of all patches and labels 

all_patches = [split_image(img,7,7) for img, _ in train_dataset] 
all_patches = torch.stack(all_patches)
all_labels = torch.tensor([label for _, label in train_dataset])

print (type(all_patches))
print (type(all_labels))
print (all_patches.shape)
print (all_labels.shape)

def patch_to_vector(dataset='Tensor'):
    '''
    convert each patch into a vector of size 49     
    '''
    dataset = dataset
    return dataset.view(60000,16,-1)

dataset_vectors = patch_to_vector(all_patches)
print(dataset_vectors.shape)
print(type(dataset_vectors))

# create linear nn to project patches to 64 dimension vectors 
patch_size = 7
embed_dim = 64
patch_dim = patch_size * patch_size

class PatchProjector(torch.nn.Module):
    
    def __init__(self, patch_dim, embed_dim):
    
        super(PatchProjector, self).__init__()
        self.linear = nn.Linear(patch_dim, embed_dim)
        
    def forward(self, x):
        '''
        Forward pass through the linear layer
        to project patches to 64 dimension vectors
        
        '''
        x = self.linear(x)
        return x


patch_projector = PatchProjector(patch_dim, embed_dim)
example_patches = dataset_vectors[0].float()
example_output = patch_projector(example_patches)
print("Example output shape:", example_output.shape) # output should be (16, 64)

dataset_embedded = PatchProjector(patch_dim, embed_dim)(dataset_vectors)
print("dataset_embedded shape:", dataset_embedded.shape)


# prepend CLR, random matrix, 1 x 64, 

# extract dimensions of embedded dataset 
B, N, D = dataset_embedded.shape
# make a random matrix of size (1 x 1 x 64)
CLS_token = nn.Parameter(torch.zeros(1,1, D))
nn.init.trunc_normal_(CLS_token, std=0.02) # from paper

# expand CLS_tokens to match the batch size 
CLS_token = CLS_token.expand(B,-1,-1)

# concatanate CLS_token to the embedded dataset
mnist_vctrs_CLS = torch.cat([CLS_token, dataset_embedded], dim=1)

# print the shape of the new dataset
print("mnist_vctrs_CLS shape:", mnist_vctrs_CLS.shape) # should be 60000 x 17 x 64


# add positional encoding to the patch embeddings and CLS token
print ("Dimensions of embedded dataset befor CLS:", B, N, D) # 60000 x 16 x 64

# Add positional encoding to the patch embeddings and CLS token

# Create positional embeddings (learnable)
position_embed = nn.Parameter(torch.zeros(1, N + 1, D))  # (1, 17, 64)
nn.init.trunc_normal_(position_embed, std=0.02)

# Add positional encoding to visual transformer input
# The first position corresponds to the CLS token, followed by the patches
# The positional encoding is added to the CLS token and patch embeddings
# The shape of position_embed is (1, 17, 64) to match the shape of mnist_vctrs_CLS
Visual_Transformer_input = mnist_vctrs_CLS + position_embed 
print("Visual_Transformer_input shape:", Visual_Transformer_input.shape) # (60000, 17, 64)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.linear1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), # why by 4?
            nn.GELU(), # why gelu
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        """Forward pass of the encoder block."""
        # x shape: (B, N, D) where B is batch size, N is number of patches + 1 (CLS token), D is embedding dimension
        x_res1 = x
        x = self.linear1(x)
        x, _ = self.attention(x, x, x)  # Self-attention
        x = x + x_res1             # Residual connection 1

        x_res2 = x
        x = self.linear2(x)
        x = self.mlp(x)
        x = x + x_res2             # Residual connection 2
        return x
    
class VisualTransformer(nn.Module):
    """Visual Transformer for the MNIST dataset."""
    def __init__(self, embed_dim, num_heads, num_layers, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Forward pass of the Visual Transformer."""
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        return self.head(x[:, 0, :])  # Use CLS token for classification    

# Example with first 128 images:
B = 128
test_x = Visual_Transformer_input[:B]           # shape: (128, 17, 64)
test_labels = all_labels[:B]     # shape: (128,)

num_heads = 4
num_layers = 3  

model = VisualTransformer(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)

out = model(test_x)              # Should be (128, 10)
print(out.shape)
print(out[0])  # Print the output for the first image























    
# create an instance of the PatchProjector


# Example usage
# project the first 16 patches from the first image 

print("Example patches shape:", example_output.shape)


#### Training the PatchProjector ###

# define a loss function and optimizer 
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(patch_projector.parameters(), lr=0.001)
# Training Loop 
num_epochs = 5
for epoch in range(num_epochs):
    for i in range(dataset_img_ptch_grps.shape[0]):
        # get the patches for the current image
        patches = dataset_img_ptch_grps[i].float()
        
        # forward pass 
        outputs = patch_projector(patches)
        # calculate loss 
        loss = criterion(outputs, patches)
        # backward pass and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        # save the model every 100 iterations
        if (i+1) % 100 == 0:
            torch.save(patch_projector.state_dict(), f'patch_projector_epoch{epoch+1}_iter{i+1}.pth')
        # print loss    
            print(f'epoch {epoch+1}, iteration {i+1}, loss: {loss.item()}')