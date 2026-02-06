import torch
from generator import Generator
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator(100).to(device)
G.load_state_dict(torch.load("../checkpoints/G_99.pth"))
G.eval()

z = torch.randn(25, 100, 1, 1, device=device)
fake_images = G(z)

save_image(fake_images, "generated.png", nrow=5, normalize=True)
