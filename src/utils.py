import torch

def retain(*args):
    for arg in args:
        arg.retain_grad()