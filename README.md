# Reproducing-and-Enhancing-LLM2CLIP-for-Cross-Modal-Artwork-Retrieval

🚧 This repository is still under construction. More files and documentation will be added soon.

This project is part of my Master's thesis and builds upon the [LLM2CLIP](https://github.com/microsoft/LLM2CLIP) framework. While the training pipeline follows the original LLM2CLIP implementation, this repository contains additional components for WikiArt dataset and retrieval evaluation.

## 📁 Folder Overview

- `dataset/`: Contains WikiArt-based evaluation sets and links to the image data.
- `evaluation/`: Scripts for embedding extraction and retrieval evaluation on different test settings.
- `knowledge_injection/`: Contains training scripts and style-specific training data for knowledge injection via InfoNCE loss. (⚠️ Note: The training environment (e.g., dependencies, configuration files) is not yet fully organized. )

Additional training scripts can be found in the official [LLM2CLIP repository](https://github.com/microsoft/LLM2CLIP), which this work directly extends.

