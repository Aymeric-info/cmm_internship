import torch

# chosing device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# training parameters
time_emb_dim = 128
base_channels = 32
time_steps = 500
epochs = 100
finetuning_epochs = 1000

r = 4   # rank for Low Rank Adaptation

# inference
start_time = 320