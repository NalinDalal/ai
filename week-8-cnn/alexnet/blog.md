# Building an Image Classifier with CNNs: AlexNet & ResNet

## Introduction
Convolutional Neural Networks (CNNs) have revolutionized image classification. AlexNet, introduced in 2012, marked a breakthrough in deep learning for computer vision, and modern variants like ResNet continue to push the boundaries. In this blog, we’ll walk through the key ideas, architecture, and practical steps to build your own image classifier using CNNs.

## Why CNNs?
Traditional neural networks struggle with image data due to the high dimensionality and spatial structure. CNNs solve this by using convolutional layers that learn local patterns, making them efficient and powerful for visual tasks.

## AlexNet: The Game Changer
AlexNet was trained on the massive ImageNet dataset (15 million images, 22,000 categories). Its innovations included:
- Deep architecture: 5 convolutional + 3 fully-connected layers
- ReLU activation for faster training
- Dropout and data augmentation to reduce overfitting
- GPU training for scalability
- Local Response Normalization (LRN) for improved generalization

### Key Results
AlexNet achieved top-1 and top-5 error rates of 37.5% and 17.0% on ILSVRC-2010, outperforming previous methods by a large margin.

## Modern CNNs: ResNet
ResNet introduced residual connections, allowing networks to go deeper without vanishing gradients. This enables even better accuracy and stability.

## Building Your Own Classifier
Here’s a step-by-step outline:
1. **Choose a dataset**: CIFAR-10, ImageNet, or your own images
2. **Define the CNN architecture**: Use AlexNet, ResNet, or a custom model
3. **Data augmentation**: Apply transformations like flips, rotations, and color changes
4. **Training**: Use GPU acceleration, dropout, and batch normalization
5. **Evaluation**: Measure accuracy, confusion matrix, and visualize predictions

## Example: ResNet-18 on CIFAR-10
Our project implements a ResNet-18 classifier for CIFAR-10. The pipeline includes:
- Data loading and preprocessing
- Model definition (see init.py)
- Training loop with optimizer and loss function
- Evaluation on test data

## Tips for Success
- Use ReLU activations for faster convergence
- Apply dropout and data augmentation to prevent overfitting
- Leverage GPU for efficient training
- Experiment with deeper models (ResNet, VGG, etc.)

## Conclusion
CNNs are the foundation of modern image classification. By understanding architectures like AlexNet and ResNet, and applying best practices in training, you can build powerful classifiers for a wide range of visual tasks.

---

**Project Code:** [init.py](./alexnet-paper/init.py)

**References:**
- [AlexNet Paper](http://papers.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Dive into Deep Learning: AlexNet](https://d2l.ai/chapter_convolutional-modern/alexnet.html)
