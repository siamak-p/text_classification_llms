# text_classification_llms

This project demonstrates two different approaches to classify text into emotional categories using the [DistilBERT](https://huggingface.co/distilbert-base-uncased) language model and an **emotion-labeled dataset**. The implementation is done in **Jupyter Notebook**.



##  Objectives

- Load and preprocess an **emotion classification dataset**.
- Use **DistilBERT** embeddings for feature extraction.
- Compare two classification approaches:
  1. **Feature-based classification** using traditional ML models (e.g., Logistic Regression, SVM).
  2. **Fine-tuning DistilBERT** using Hugging Face Transformers.



## Approaches

### 1. Feature Extraction + ML Classifier
- **Steps**:
  - Tokenize input using DistilBERT tokenizer.
  - Extract hidden-state embeddings from DistilBERT.
  - Train a traditional classifier (e.g., Logistic Regression, RandomForest).

- Pros: Fast to train and interpretable.
- Cons: Does not leverage full potential of the LLM.


### 2. Fine-Tuning DistilBERT
- **Steps**:
  - Use Hugging Face `Trainer` API.
  - Add a classification head on top of DistilBERT.
  - Fine-tune the model on emotion labels.
  - Evaluate performance (accuracy, F1-score).

- Pros: Higher accuracy, end-to-end optimization.
- Cons: Requires more computational resources.



## Dataset

- The dataset contains sentences or phrases labeled with emotions such as: `joy`, `anger`, `sadness`, `fear`, etc.
- You can use any public dataset like:
  - [Emotion Dataset from HuggingFace Datasets](https://huggingface.co/datasets/dair-ai/emotion)
  - Or upload your own CSV to the `data/` folder.



## Running the Notebook

### 1. Clone this repository

git clone https://github.com/siamak-p/text_classification_llms.git

cd text_classification_llms

### 2. Install requirements and run jupyter notebook

pip install transformers datasets scikit-learn pandas matplotlib jupyter
run jupyter notebook

