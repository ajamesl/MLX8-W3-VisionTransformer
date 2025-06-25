#
#
#
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import random
import einops # for tensor re-arrangement


#
#
#
torch.manual_seed(47)
random.seed(47)


#
#
#
class Combine(torch.utils.data.Dataset):
  def __init__(self, train=True):
    super().__init__()
    # a transform that converts images to tensors and normalizes them
    self.tf = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
    # a dict that maps digits and special tokens (<start> <end>) to integer labels 
    self.tk = { '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 's': 10, 'e': 11 }
    # loads the MNIST dataset
    self.ds = tv.datasets.MNIST(root='.', train=train, download=True)
    # a transform to convert tensors back to PIL images for visualization later
    self.ti = tv.transforms.ToPILImage()
    # number of samples in the MNIST dataset
    self.ln = len(self.ds)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    idx = random.sample(range(self.ln), 4)
    store = [self.ds[i][0] for i in idx]
    label = [self.ds[i][1] for i in idx]
    tnsrs = [self.tf(img) for img in store]
    stack = torch.stack(tnsrs, dim=0).squeeze()
    combo = einops.rearrange(stack, '(h w) ph pw -> (h ph) (w pw)', h=2, w=2, ph=28, pw=28)
    patch = einops.rearrange(combo, '(h ph) (w pw) -> (h w) ph pw', ph=14, pw=14)
    label = [10] + label + [11]
    return combo, patch, torch.tensor(label)


#
#
#
if __name__ == "__main__":

  ds = Combine()
  cmb, pch, lbl = ds[0]
  print('lbl', lbl) # [2, 4, 9, 7]
  print('cmb', cmb.shape)
  print('pch', pch.shape)
  plt.imshow(ds.ti(cmb)); plt.show()
  plt.imshow(ds.ti(einops.rearrange(pch, 'p h w -> h (p w)'))); plt.show()

  pass