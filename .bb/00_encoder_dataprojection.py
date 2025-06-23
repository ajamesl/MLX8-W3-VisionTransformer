import torch 
from torchvision import datasets, transforms


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# PSEUDOCODE 
# split images into 4 x 4 = 7,7,7,7 
print (train_dataset.data.shape)
print (type(train_dataset.data))

def split_images(images, height, width):
    
    """
    Split original dataset imaghes into patches
    Takes images of single layer, 28 x 28 
    Grayscale Images which have been normalized
    
    Arguments: 
    images -- tensor of shape (N, 1, 28, 28) where N is the number of images
    height - the required height of the patches
    width - the required width of the patches
    
    """    

    patch_height = height
    patch_width = width

    patches = images.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    
    return patches.contiguous().view(-1, 1, patch_height, patch_width) 

dataset_patches = split_images(train_dataset.data.unsqueeze(1), 7, 7)
print(dataset_patches.shape)
print(type(dataset_patches))


# Set up linear layer to feed patches through (169 x 64) outputting vector of (1 x 64)

# take first 16 patches 
# put each patch into one vector 

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


# output should be 64 dimension vectors for each patch 

