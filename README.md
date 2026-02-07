# Synthetic Crop Leaf Disease Image Generation using DCGAN

---

## Project Team

**Team Members:** Likitha, Lathif, Navadeep, Sameer, Hari Priya, Sruthika  

---

## Overview

This project focuses on generating realistic synthetic images of diseased crop leaves using a **Deep Convolutional Generative Adversarial Network (DCGAN)**. The generated images are used to augment imbalanced agricultural datasets, improving the performance of plant disease detection models.

---

## Motivation

Agricultural disease datasets are often highly imbalanced, with many healthy leaf images but very few samples of rare diseases such as rust, blight, or mildew. Collecting and labeling field data is expensive, time-consuming, and region-specific. Models trained on such datasets perform poorly on rare disease classes. This project leverages DCGANs to generate high-quality synthetic disease images, reducing data scarcity and improving model fairness.

---

## Dataset

The project uses publicly available multi-crop leaf disease image datasets containing **RGB images** of healthy and diseased leaves across different crops such as:

- Tomato  
- Potato  
- Grape  
- Maize  
- Wheat  

Images are resized and normalized for GAN training.

---

## Approach

A **DCGAN architecture** is implemented with two main components:

### Generator
- Takes random noise as input  
- Uses transposed convolution layers  
- Generates synthetic crop leaf disease images  

### Discriminator
- Takes real or synthetic images as input  
- Classifies images as real or fake  
- Uses convolutional layers with LeakyReLU activation  

Both networks are trained adversarially using **binary cross-entropy loss** and the **Adam optimizer**.

---

## Evaluation

Model performance is evaluated using:

- Visual inspection of generated images  
- Monitoring generator and discriminator losses  
- Comparing disease classifier performance with and without synthetic images  
- Measuring accuracy and F1-score for rare disease classes  

---

## Applications

- Crop disease detection systems  
- Agricultural advisory platforms  
- Dataset augmentation for agri-AI research  
- Educational tools for plant pathology  

---

## Conclusion

This project demonstrates that **DCGAN-based synthetic image generation** effectively reduces dataset imbalance in agricultural disease datasets. The generated images improve disease classification performance and provide a scalable, cost-effective solution for agricultural AI applications.
