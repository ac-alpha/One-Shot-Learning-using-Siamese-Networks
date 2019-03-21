import torch
import torchvision
import torchvision.transforms as transforms
import glob

def downloadOmniglot(train = True):
	
	dataset = torchvision.datasets.Omniglot(root = "./data", 
		background = train, 
		download = True, 
		transform = transforms.ToTensor())

# downloadOmniglot()


