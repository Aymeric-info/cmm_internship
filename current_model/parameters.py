import torch

# chosing device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

base_weights_path = "parameters/ddpm_weights_normal.pth"

def get_lora_save_path(anomaly_idx):
    return f"parameters/ddpm_weights_lora_anomaly_{anomaly_idx}.pth"

# training parameters
time_emb_dim = 128
base_channels = 32
time_steps = 500
epochs = 50
finetuning_epochs = 120

r = 16   # rank for Low Rank Adaptation
finetuning_batch_size = 2

# inference
start_time = 320