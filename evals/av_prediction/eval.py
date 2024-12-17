import os
import torch
import torch.nn as nn

att = True


# -- Load pretrained Encoder and Predictor (frozen)
encoder = ...

predictor = ...


if att:
    probe = att_probe()
else:
    probe = lin_probe()





def run_one_epoch():

    return