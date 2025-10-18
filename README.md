# Xorl

Xorl is simple and high-performance distributed training framework for large language models, based on [Xorl](https://github.com/xorl-org/Xorl).


## Overview

Xorl is a distributed training framework for large language models, built on top of [Xorl](https://github.com/xorl-org/Xorl). It provides simple, high-performance training across multiple GPUs and nodes.

### Key Features

- High-performance distributed training
- Modular and flexible design
- Linear training scripts (no rigid trainer classes)
- Native PyTorch integration
- Easy scaling from single GPU to clusters

### Design Principles

Xorl follows Xorl's core principles:

- **Flexibility**: Modular components that can be easily customized or replaced
- **Trainer-free**: Linear training scripts instead of rigid trainer classes
- **Model-agnostic**: Works with any model architecture
- **PyTorch-native**: Leverages PyTorch's native functions for compatibility and performance