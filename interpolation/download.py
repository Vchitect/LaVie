# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os


pretrained_models = {''}


def find_model(model_name):
    """
    Finds a pre-trained model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint

