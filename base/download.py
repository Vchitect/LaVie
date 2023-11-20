import torch
import os


def find_model(model_name):
    """
    Finds a pre-trained model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        print('Ema existing!')
        checkpoint = checkpoint["ema"]
    return checkpoint
