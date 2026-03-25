## 📌 Multi-label Text Classification with RuBERT (UGC Moderation)

This project solves a **multi-label text classification** problem on user-generated content (UGC) using a fine-tuned **RuBERT** model.

The goal is to assign **multiple relevant categories (out of 50)** to each text response.

---

## 🧠 Problem Description

Each input sample consists of:

* Free-form user text
* Selected tags from a survey

Each sample may belong to **multiple classes simultaneously**, making this a **multi-label classification task**.

---

## 🔬 Training Pipeline

### 1. Domain Adaptation (MLM)

* Additional pretraining using Masked Language Modeling
* Helps adapt RuBERT to domain-specific text

### 2. Multi-label Fine-tuning

* Loss: `BCEWithLogitsLoss`
* Optimizer: AdamW
* Scheduler: Linear warmup

### 3. Two-stage Training

* Stage 1: Train classifier head only
* Stage 2: Fine-tune full model

---

## 📊 Results

| Metric   | Score     |
| -------- | --------- |
| F1 Score | **0.743** |
| ROC-AUC  | **0.888** |
| Accuracy | 0.571     |

---

## 📂 Dataset

* **train.csv**

  * text
  * tags
  * 50 target labels

* **test.csv**

  * text
  * tags

* **trends_description.csv**

  * class descriptions

---

## 🧹 Preprocessing

* Lowercasing
* Punctuation cleaning
* Emoji removal
* Tag normalization (mapping to readable text)

---

## 📂 Project Structure

```
.
├── fine_tuned_rubert.ipynb   # Full pipeline
└── README.md
```

---

## ⚙️ Installation

```bash
pip install torch transformers pandas scikit-learn tqdm
```

---

## ▶️ Usage

```bash
jupyter notebook fine_tuned_rubert.ipynb
```

Run all cells to:

* preprocess data
* pretrain model (MLM)
* train classifier
* evaluate results

---

## 💡 Key Insights

* MLM pretraining significantly improves domain adaptation
* Using last 4 hidden states boosts performance vs CLS-only
* Multi-label tasks require **sigmoid + BCE**, not softmax

---

## 🚀 Future Improvements

* Per-class threshold tuning
* Handling class imbalance
* Using larger LLMs

---

