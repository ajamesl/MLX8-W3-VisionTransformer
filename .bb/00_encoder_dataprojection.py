import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
%matplotlib inline

import sys
print(sys.executable)
import numpy
print(numpy.__version__)


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# PSEUDOCODE 
# split images into 4 x 4 = 7,7,7,7 cd # split images into patches of size 7 x 7
# create linear layer to project patches to 64 dimension vectors
img, label = train_dataset[0]
print(img.shape)    # e.g., torch.Size([1, 28, 28])
print(type(label))

print (train_dataset.data.shape)
print (type(train_dataset.data))

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


# Set up linear layer to feed patches through (169 x 64) outputting vector of (1 x 64)

# take first 16 patches 
# put each patch into one vector 

all_patches = [split_image(img,7,7) for img, _ in train_dataset] 
all_patches = torch.stack(all_patches)
all_labels = torch.tensor([label for _, label in train_dataset])

print (type(all_patches))
print (type(all_labels))
print (all_patches.shape)
print (all_labels.shape)

def patch_to_vector():
    '''
    convert each patch into a vector of size 49  
    QUESTION: do we need to keep the labels? 
       
    '''
    return dataset_patches.view(dataset_patches.shape[0], -1)

dataset_vectors = patch_to_vector()
print(dataset_vectors.shape)
print(type(dataset_vectors))

# split dataset vectors into groups iof 16 patches ( one for each image)
def split_into_groups(vectors, group_size):
    """
    split the dataset vectors into groups according to the original images

    Args:
        vectors (_type_): _description_
        group_size (_type_): _description_
    """
    num_groups = vectors.shape[0] // group_size
    return vectors.view(num_groups, group_size, -1) 

dataset_img_ptch_grps = split_into_groups(dataset_vectors, 16) 

print(dataset_img_ptch_grps.shape)
print(type(dataset_img_ptch_grps))

# create linear nn to project patches to 64 dimension vectors 
class PatchProjector(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
    
        super(PatchProjector, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        '''
        Forward pass through the linear layer
        to project patches to 64 dimension vectors
        
        '''
        x = self.linear(x)
        return x


patch_projector = PatchProjector(49, 64)
example_patches = dataset_img_ptch_grps[0].float()
example_output = patch_projector(example_patches)
print("Example output shape:", example_output.shape)

# prepend CLR, random matrix, 1 x 64, 

























    
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