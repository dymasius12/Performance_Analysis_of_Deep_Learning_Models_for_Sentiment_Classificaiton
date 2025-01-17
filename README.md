#Performance Analysis of Deep Learning Models for Sentiment Classification Analysis on IMDB Dataset

## Overview
This project evaluates the performance of six deep learning models (FFNNs, CNNs, LSTMs, BiLSTMs) for sentiment classification using the IMDB dataset. The study investigates the impact of optimizers (SGD, Adam, Adagrad) and training epochs on model performance.

## Dataset
We are using a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. See the README file contained in the release for more details. 
Credits: https://ai.stanford.edu/~amaas/data/sentiment/ 

## Key Findings
1. **Adagrad Optimizer**: Best test accuracy (73.28%) and lowest test loss (0.572).
2. **20 Epochs**: Balanced generalization with a test accuracy of 70.71% and test loss of 0.593.
3. **One-layer FFNN with Adam**: Outperformed other architectures in this experiment.

## Experiment Results
### Optimizer Comparison
- **Loss and Accuracy**:
  - Evaluated the performance of SGD, Adam, and Adagrad optimizers.

![Optimizer Loss](NLP_A1/NLPA1_16JAN2025_1A_Loss.png)
![Optimizer Accuracy](NLP_A1/NLPA1_16JAN2025_1B_Acc.png)

- **Comparison**:
  - Visualizes test loss and accuracy for the optimizers.

![Optimizer Comparison](NLP_A1/NLPA1_16JAN2025_1C_Comparison.png)

### Epoch Variation
- **Loss and Accuracy**:
  - Explores the effect of different epoch configurations (5, 10, 20, 50).

![Epoch Loss](NLP_A1/NLPA1_16JAN2025_2A_Loss.png)
![Epoch Accuracy](NLP_A1/NLPA1_16JAN2025_2B_Accuracy.png)

- **Comparison**:
  - Summarizes the test loss and accuracy across different epochs.

![Epoch Comparison](NLP_A1/NLPA1_16JAN2025_2C_Comparison.png)

### Model Comparison
- **Loss and Accuracy**:
  - Compares performance of six model architectures (One-layer FFNN, Two-layer FFNN, Three-layer FFNN, CNN, LSTM, BiLSTM).

![Model Loss](NLP_A1/NLPA1_16JAN2025_3A_Loss.png)
![Model Accuracy](NLP_A1/NLPA1_16JAN2025_3B_Acc.png)

- **Training Trends**:
  - Highlights training and validation loss/accuracy trends for the models.

![Model Loss Trends](NLP_A1/NLPA1_16JAN2025_3C_Loss.png)
![Model Accuracy Trends](NLP_A1/NLPA1_16JAN2025_3D_Accuracy.png)

- **Comparison**:
  - Summarizes test loss and accuracy for each model.

![Model Comparison](NLP_A1/NLPA1_16JAN2025_3E_Comparison.png)


## Requirements
- Python 3.8+
- TensorFlow, Keras, Matplotlib, NumPy, Pandas

Install dependencies:
```
pip install -r requirements.txt
```
