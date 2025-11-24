# 📘 Kindle Reviews Sentiment Analysis using BiLSTM

End-to-end NLP pipeline for binary sentiment classification on Amazon Kindle reviews using TensorFlow, BiLSTM, Embedding layers, and a fully balanced training workflow. This project delivers a robust, production-grade text classifier with strong evaluation metrics and clear explainability.

---

## 🚀 Project Overview

This project builds a sentiment classifier that predicts whether a Kindle review is **positive (>3 stars)** or **negative (≤3 stars)**.

The workflow includes:

* Data cleaning & preprocessing
* Tokenization + padded sequences
* Class balancing (upsampling)
* BiLSTM neural architecture
* Model training with callbacks
* Evaluation using accuracy, ROC-AUC, AUPRC
* Misclassification analysis

Final performance: **83% accuracy**, stable ROC-AUC, and strong calibration.

---

## 📂 Repository Structure

```
📁 kindle-sentiment-bilstm
 ├── all_kindle_review.csv
 ├── sentiment_model.ipynb
 ├── best_rnn_binary.h5
 ├── tokenizer_config.pkl (optional)
 ├── README.md
 └── /plots
```

---

## 🧼 Data Preprocessing

* Lowercasing
* Removing HTML tags & noise
* Cleaning summary + review text
* Handling null values
* Binary labeling rule:

  * **rating ≤ 3 → 0 (Negative)**
  * **rating > 3 → 1 (Positive)**

The text is cleaned using regex-based normalization and combined into a unified `text` column.

---

## 🧱 Model Architecture

```
Embedding (30k vocab, 128 dims)
BiLSTM (128 units)
Dropout (0.3)
Dense (64, ReLU)
Dropout (0.2)
Dense (1, Sigmoid)
```

Optimized for stability, low generalization error, and performance on imbalanced datasets.

---

## 📊 Model Performance (Test Set)

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **0.8319** |
| Precision | 0.83       |
| Recall    | 0.83       |
| F1 Score  | 0.83       |
| ROC-AUC   | 0.8319     |
| AUPRC     | 0.7756     |

Balanced results across both sentiment classes.

---

## 📈 Evaluation Visuals

Included in the repo:

* ROC Curve
* Precision–Recall Curve
* Calibration Curve
* Misclassified sample analysis

---

## 🧪 Training Strategy

* Upsampling used to equalize classes
* EarlyStopping (patience=3)
* ModelCheckpoint (best weights saved)
* Adam optimizer with LR = 2e-4
* Batch size: 64
* Epochs: 12

---

## 🔍 Error Analysis

The script prints:

* True label
* Predicted label
* Predicted probability
* Reconstructed text from padded sequences

Helps understand where the model struggles.

---

## 🛠 Tech Stack

* Python 3.x
* TensorFlow / Keras
* Scikit-learn
* NLTK
* NumPy, Pandas
* Matplotlib

---

## 📦 Running the Project

```
pip install -r requirements.txt
python sentiment_training.py
```

Or open the notebook:

```
sentiment_model.ipynb
```

---

## ⭐ Future Enhancements

* GRU / Transformer Encoder-based models
* Add pretrained embeddings (GloVe / Word2Vec)
* Build a FastAPI-based API layer
* Add LIME/SHAP interpretability

---

## 🤝 Contributions

Open to PRs and enhancements for deployment, optimization, or feature extension.

---

## 🏁 Summary

This repository showcases a clean, scalable NLP pipeline with strong performance metrics and a polished engineering workflow—ideal for portfolio display or real-world deployment.
