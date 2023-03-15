import os
import torch
import torch.nn as nn
import torch.optim as optim

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt"))

def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint
