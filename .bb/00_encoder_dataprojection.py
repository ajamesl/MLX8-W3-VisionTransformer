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
print("Position embed shape:", position_embed.shape)  # should be (1, 17, 64)

# Add positional encoding to visual transformer input
# The first position corresponds to the CLS token, followed by the patches
# The positional encoding is added to the CLS token and patch embeddings
# The shape of position_embed is (1, 17, 64) to match the shape of mnist_vctrs_CLS
# Dimensions of size 1 in the smaller tensor are stretched to match the corresponding size in the larger tensor
# so the same positional encoding is added identically across the entire batch 
visual_transformer_input = mnist_vctrs_CLS + position_embed 


print("visual_transformer_input shape:", visual_transformer_input.shape) # (60000, 17, 64)

# demonstrate the difference between visual_transformer_input and mnist_vctrs_CLS
print("Visual Transformer Input vs MNIST Vectors CLS:")
print("Visual Transformer Input:", visual_transformer_input[0])  # First image with CLS
print("MNIST Vectors CLS:", mnist_vctrs_CLS[0])



# build self attention module

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # B, N, D = batch size, number of tokens, emb dim
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim) 
        # Applies a linear transformation to the input tensor to produce the query tensor Q using learned weights W_q
        self.k_proj = nn.Linear(embed_dim, embed_dim) 
        # Applies a linear transformation to the input tensor to produce the query tensor K using learned weights W_k
        self.v_proj = nn.Linear(embed_dim, embed_dim) 
        # Applies a linear transformation to the input tensor to produce the query tensor V using learned weights W_v
        
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """Forward pass of the self-attention module."""
        # x: B, N, D
        B, N, D = x.shape # B: batch size, N: number of tokens (patches + CLS), D: embedding dimension
        
        Q = self.q_proj(x)  # Query: B, N, D
        K = self.k_proj(x)  # Key: B, N, D
        V = self.v_proj(x)  # Value: B, N, D
        
        # Reshape Q, K, V for multi-head attention, transpose switches the dimensions 
        # to optermise for multi-head computations
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1,2)  # (B, N, num_heads, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1,2)  # (B, N, num_heads, head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1,2)  # (B, N, num_heads, head_dim)
        
        # compute attention scores
        attn_scores = Q @ K.transpose(-2, -1) * (self.embed_dim ** -0.5)  # (B, N, N),  @ sign is matrix multiplacation
        attn_weights = attn_scores.softmax(dim=-1)  # (B, N, N)

        # apply attention weights to values
        out = attn_weights @ V  # (B, N, D)
        out = out.transpose(1,2).contiguous().view(B,N, self.embed_dim)
        return self.final_linear(out)

# test self attention module 

attn = SelfAttention(embed_dim=64, num_heads=8)
x = torch.randn(2, 16, 64)  # (batch, tokens, embed dim)

out = attn(x)
print(out.shape)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, num_heads)
        #self.attention2 = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), 
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        """Forward pass of the encoder block."""
        # x shape: (B, N, D) where B is batch size, N is number of patches + 1 (CLS token), D is embedding dimension
        x_res1 = x
        x = self.norm1(x)
        x= self.attention(x) # Self-attention
        x = x + x_res1             # Residual connection 1

        x_res2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res2             # Residual connection 2
        return x
   
# test transformerencoder module 
transformer = TransformerEncoder(embed_dim=64, num_heads=8)
x = torch.randn(2, 16, 64)  # (batch, tokens, embed dim)

out = transformer(x)
print(out.shape)
   

class VisualTransformer(nn.Module):
    """Visual Transformer for the MNIST dataset."""
    def __init__(self, embed_dim, num_heads, num_layers, num_classes=10):
        super().__init__()
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Forward pass of the Visual Transformer."""
        for layer in self.encoder:
            x = self.layer_norm(x)
        x = self.layer_norm(x)
        return self.mlp_head(x[:, 0, :])  # Use CLS token for classification    


# to do 
# draw diagram in miro to clarify logic of architecture 
# exact functionality of each block / layer 
# add training loop and run on local computer 
# evaluate the model and test its peformance 
# compare to CNN 









# Example with first 128 images:
B = 128
test_x = visual_transformer_input[:B]           # shape: (128, 17, 64)
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