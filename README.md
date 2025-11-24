# Kindle-Reviews-Sentiment-Analysis-using-BiLSTM
End-to-end NLP pipeline for binary sentiment classification on Amazon Kindle reviews using TensorFlow, BiLSTM, Embedding, and a fully-balanced training setup. This project delivers a robust text-classification system with clean preprocessing, model training, evaluation, and error analysis — tuned for operational excellence and scalable deployment.

🚀 Project Overview

This repo demonstrates how to build a sentiment classifier that predicts whether a Kindle review is positive (>3 stars) or negative (≤3 stars).

The full pipeline covers:

🧹 Data cleaning & preprocessing

🧬 Text tokenization + padded sequences

⚖️ Upsampling to handle class imbalance

🧠 BiLSTM-based deep learning architecture

📊 Model performance metrics: Accuracy, ROC-AUC, AUPRC

🔍 Misclassification analysis for model insights

The system ships with 83% accuracy, solid ROC-AUC performance, and deploy-ready training assets like checkpoints & callbacks.

📂 Repository Structure
📁 kindle-sentiment-bilstm
 ├── all_kindle_review.csv         # Dataset
 ├── sentiment_model.ipynb         # Full training notebook
 ├── best_rnn_binary.h5            # Saved best model weights
 ├── tokenizer_config.pkl          # (optional) Tokenizer object
 ├── README.md                     # This file
 └── /plots                        # ROC, PR curve, calibration plots

🧼 Data Preprocessing Workflow

The pipeline applies a structured text-cleaning strategy:

Lowercasing

Removing HTML & non-alphanumeric noise

Combining summary + reviewText

Null removal

Labeling rule:

rating <= 3 → 0 (Negative)

rating > 3 → 1 (Positive)

A custom cleaning function ensures enterprise-grade consistency across text inputs.

🧱 Model Architecture

A lean yet powerful sequential architecture optimized for textual signal extraction:

Embedding (30k vocab, 128 dims)
BiLSTM (128 units)
Dropout (0.3)
Dense (64, ReLU)
Dropout (0.2)
Dense (1, Sigmoid)


Hyperparameters optimized for:

Lower generalization error

Stability across imbalanced datasets

Minimal overfitting (via EarlyStopping + ModelCheckpoint)

📊 Model Performance (Test Set)
Metric	Score
Accuracy	0.8319
Precision	0.83
Recall	0.83
F1 Score	0.83
ROC-AUC	0.8319
AUPRC	0.7756

The model demonstrates strong calibration and balanced class performance.

📈 Evaluation Visuals

The project includes:

ROC Curve — AUC ~0.83

Precision-Recall Curve — AUPRC ~0.77

Calibration Curve — probability reliability

Error Analysis — top misclassified samples for root-cause investigation

🧪 Training Strategy
Balanced Learning

Applied upsampling to ensure equal representation of both sentiment classes in training.

Callbacks Used

EarlyStopping (patience=3)

ModelCheckpoint (best weights)

Training Statistics

Epochs: 12

Batch size: 64

Optimizer: Adam (lr=2e-4)

🔍 Error Analysis

The script prints out misclassified rows along with:

True label

Predicted label

Model probability

Reconstructed text from padded sequence

This creates full transparency for evaluating model blind spots.

🛠 Technologies Used

Python 3.x

TensorFlow / Keras

Scikit-Learn

NLTK

NumPy + Pandas

Matplotlib

📦 How to Run
pip install -r requirements.txt
python sentiment_training.py


Or open sentiment_model.ipynb in Jupyter/Colab.

🤝 Contributing

PRs are welcome. If you want to scale this model into production-grade microservices (FastAPI + Docker + AWS), feel free to collaborate.

⭐ Future Enhancements

Switch to GRU or Transformer Encoder

Integrate GloVe/Word2Vec embeddings

Deploy REST API using FastAPI

Add explainability using LIME/SHAP

🏁 Final Thoughts

This repository is a streamlined, data-first implementation showcasing clean NLP engineering pipelines with practical deep-learning insights. Ideal for interview prep, portfolio display, or production prototyping.
