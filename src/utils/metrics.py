"""
Metrics module for GAN evaluation
"""

def simple_gan_score(generator_loss, discriminator_loss):
    """
    Simple heuristic metric to observe GAN balance.
    Smaller difference indicates stable training.
    """
    return abs(generator_loss - discriminator_loss)


def fid_score(real_images=None, fake_images=None):
    """
    Placeholder for Fréchet Inception Distance (FID).
    Actual implementation requires feature extraction
    using a pretrained network.
    """
    print("FID score calculation placeholder")
    return 0.0
