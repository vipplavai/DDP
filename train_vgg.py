import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms


def setup(rank, world_size):
   os.environ['MASTER_ADDR'] = '172.26.112.36'
   os.environ['MASTER_PORT'] = '29500'
   print(f"Setting up the distributed environment: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}, Rank={rank}, World Size={world_size}")
   dist.init_process_group("nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(0)
   print("Distributed environment setup complete.")


def cleanup():
   print("Cleaning up the distributed environment.")
   dist.destroy_process_group()


class TinyVGG(nn.Module):
   def __init__(self):
       super(TinyVGG, self).__init__()
       self.features = nn.Sequential(
           nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
           nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
           nn.MaxPool2d(2, 2),
           nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
           nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
           nn.MaxPool2d(2, 2),
       )
       self.classifier = nn.Linear(128 * 8 * 8, 10)


   def forward(self, x):
       x = self.features(x)
       x = torch.flatten(x, 1)
       x = self.classifier(x)
       return x


def train(rank, world_size):
   setup(rank, world_size)
   model = TinyVGG().cuda()
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   model = DDP(model, device_ids=[0])
   print(f"Model and DDP setup complete on rank {rank}.")


   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
   print("Training data loaded.")


   for epoch in range(1):  # loop over the dataset multiple times
       print(f"Starting Epoch {epoch+1} on Rank {rank}.")
       for i, data in enumerate(trainloader, 0):
           inputs, labels = data[0].cuda(0), data[1].cuda(0)
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           if i % 100 == 0:
               print(f"Rank {rank}, Epoch {epoch+1}, Batch {i}, Loss: {loss.item()}")
       print(f"Epoch {epoch+1} complete on Rank {rank}. Loss: {loss.item()}")


   cleanup()


if __name__ == "__main__":
   import sys
   rank = int(sys.argv[1])
   world_size = int(sys.argv[2])
   train(rank, world_size)
