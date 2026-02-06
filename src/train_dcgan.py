import torch
import yaml
from tqdm import tqdm
from data_loader import get_dataloader
from dcgan_model import build_dcgan
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image


with open("../configs/train_config.yaml") as f:
    train_cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = train_cfg["latent_dim"]
epochs = train_cfg["epochs"]
batch_size = train_cfg["batch_size"]

dataloader = get_dataloader(
    "../Data/Real/Train",
    64,
    batch_size
)

G, D = build_dcgan(latent_dim, device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=train_cfg["lr"], betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=train_cfg["lr"], betas=(0.5, 0.999))

for epoch in range(epochs):
    for real, _ in tqdm(dataloader):
        real = real.to(device)
        batch = real.size(0)

        # Labels
        real_labels = torch.ones(batch, device=device) * 0.9
        fake_labels = torch.zeros(batch, device=device)

        # Train Discriminator
        z = torch.randn(batch, latent_dim, 1, 1, device=device)
        fake = G(z)

        loss_D = (
            criterion(D(real), real_labels) +
            criterion(D(fake.detach()), fake_labels)
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        loss_G = criterion(D(fake), real_labels)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    save_image(fake[:16], f"../samples/epoch_{epoch}.png", normalize=True)
    torch.save(G.state_dict(), f"../checkpoints/G_{epoch}.pth")

    print(f"Epoch {epoch} | D Loss: {loss_D.item():.3f} | G Loss: {loss_G.item():.3f}")
