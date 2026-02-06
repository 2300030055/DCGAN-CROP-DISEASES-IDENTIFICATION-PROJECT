import torch
from generator import Generator
from discriminator import Discriminator


def build_dcgan(latent_dim, device):
    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)
    return G, D
