import matplotlib.pyplot as plt
from PIL import Image
import os


def show_sample_image(epoch=10):
    """
    Displays a generated GAN image for a given epoch
    """
    image_path = f"../samples/epoch_{epoch}.png"

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Generated Samples - Epoch {epoch}")
    plt.show()


if __name__ == "__main__":
    print("Visualization module running")
    show_sample_image(epoch=0)   # change epoch number if needed
