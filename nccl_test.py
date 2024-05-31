import os
import torch
import torch.distributed as dist

def init_process_group(rank, world_size):
	print("Setting environment variables for master node.")
	os.environ['MASTER_ADDR'] = '192.168.1.12'  # IP of the master node
	os.environ['MASTER_PORT'] = '29500'     	# Port for NCCL communication
	print(f"MASTER_ADDR set to: {os.environ['MASTER_ADDR']}")
	print(f"MASTER_PORT set to: {os.environ['MASTER_PORT']}")

	print(f"Initializing the process group for NCCL on Rank {rank}.")
	dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
	print("Process group initialized.")

def main(rank, world_size):
	init_process_group(rank, world_size)
	print(f"Setting the CUDA device for Rank {rank}.")
    
	# Set CUDA device to 0 if each machine has only one GPU
	torch.cuda.set_device(0)
	print(f"Current CUDA device: {torch.cuda.current_device()}")

	tensor = torch.ones(1, device='cuda')
	print(f"Rank {rank} - Tensor before all-reduce: {tensor.item()}")

	dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
	print(f"Rank {rank} - Tensor after all-reduce: {tensor.item()}")

	dist.destroy_process_group()
	print(f"Rank {rank} - Process group destroyed.")



if __name__ == "__main__":
	import sys
	rank = int(sys.argv[1])
	world_size = int(sys.argv[2])
	main(rank, world_size)
