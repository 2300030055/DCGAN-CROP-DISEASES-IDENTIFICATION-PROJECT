#Synthetic Crop Leaf Disease Image Generation using DCGAN
##Project Team

**Team Members:** Likitha, Lathif, Navadeep, Sameer, Hari Priya, Sruthika

---

##Overview

This project focuses on generating realistic synthetic images of diseased crop leaves using a Deep Convolutional Generative Adversarial Network (DCGAN). The generated images are used to augment imbalanced agricultural datasets, improving the performance of plant disease detection models.

---

#Motivation

Agricultural disease datasets are often highly imbalanced, with many healthy leaf images but very few samples of rare diseases such as rust, blight, or mildew. Collecting and labeling field data is expensive, time-consuming, and region-specific. Models trained on such datasets perform poorly on rare disease classes. This project leverages DCGANs to generate high-quality synthetic disease images, reducing data scarcity and improving model fairness.

--

#Dataset

The project uses publicly available multi-crop leaf disease image datasets containing RGB images of healthy and diseased leaves across different crops such as tomato, potato, grape, maize, and wheat. Images are resized and normalized for GAN training.

Example sources include PlantVillage-style leaf disease datasets.

---

##Approach

A DCGAN architecture is implemented with:

A Generator that converts random noise vectors into realistic crop leaf disease images

A Discriminator that distinguishes between real and synthetic leaf images

The model is trained adversarially using non-saturating GAN loss with Adam optimization. Separate models can be trained per disease class or on mixed disease data for generalization.

---

##Evaluation

The quality and usefulness of generated images are evaluated using:

Visual inspection of disease patterns, textures, and color realism

GAN metrics such as FID and Inception Score

Downstream evaluation using disease classification models trained with and without synthetic data

Performance improvement is measured using accuracy and F1-score, especially for rare disease classes.

---

##Applications

Data augmentation for crop disease classification models

AI-powered agricultural advisory systems

Research and teaching in plant pathology and agri-AI

Reducing dependency on expensive field data collection

---

##Conclusion

This project demonstrates that DCGAN-based synthetic image generation can effectively address data imbalance in agricultural disease datasets. By augmenting rare disease classes with realistic synthetic images, the system improves classification performance and supports scalable, cost-effective AI solutions for agriculture.
