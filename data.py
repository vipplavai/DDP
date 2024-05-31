import torch
import torchvision
import torchvision.transforms as transforms

def download_cifar10():
	# Define a transform to normalize the data
	transform = transforms.Compose([
    	transforms.ToTensor(),  # Convert images to PyTorch tensors
    	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the dataset
	])

	# Download the training data
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        	download=True, transform=transform)
    
	# Download the test data
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       	download=True, transform=transform)
    
	print("CIFAR-10 dataset downloaded.")

if __name__ == "__main__":
	download_cifar10()
