# Sentiment Analysis on IMDB Dataset

## Overview
This project evaluates the performance of six deep learning models (FFNNs, CNNs, LSTMs, BiLSTMs) for sentiment classification using the IMDB dataset. The study investigates the impact of optimizers (SGD, Adam, Adagrad) and training epochs on model performance.

## Key Findings
1. **Adagrad Optimizer**: Best test accuracy (73.28%) and lowest test loss (0.572).
2. **20 Epochs**: Balanced generalization with a test accuracy of 70.71% and test loss of 0.593.
3. **One-layer FFNN with Adam**: Outperformed other architectures in this experiment.

## Requirements
- Python 3.8+
- TensorFlow, Keras, Matplotlib, NumPy, Pandas

Install dependencies:
```
pip install -r requirements.txt
```
