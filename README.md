# Review-Rating-LSTM

A fully reproducible deep learning pipeline for predicting **1–10 star ratings** from free-form movie reviews.  
The project was developed as part of the *Dev-Acad NLP* challenge and demonstrates modern natural-language-processing (NLP) techniques with **TensorFlow 2.19 (GPU)**, spaCy and Scikit-Learn. The instructions assume you are running under **WSL 2** on Windows with an NVIDIA GPU.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Automatic data rebuild** | The `data/` folder is *not* version-controlled. If missing, the notebook unzips the raw `dev-acad-nlp` corpus and recreates the same structure automatically. |
| **Text preprocessing** | Tokenisation, stop-word removal & lemmatisation using *spaCy `en_core_web_sm`*. |
| **TextVectorization layer** | Converts raw review text into padded integer sequences using a vocabulary size of 30,000. |
| **Compact LSTM model** | A single LSTM layer is used for sequence modeling, followed by dense layers. |
| **Training utilities** | Includes EarlyStopping and ReduceLROnPlateau to improve generalisation. |
| **Single-command submission** | Generates a ready-to-upload `submission.csv` matching the competition format. |

---

## 🗂️ Project Structure

```text
.
├── LSTM_Text_Classification.ipynb   # End-to-end notebook
├── dev-acad-nlp.zip                # Raw corpus (11k train + 3k test)
├── data/                           # Auto-generated directory
│   ├── train/ trainXXXX.txt        # Cleaned texts (Latin-1)
│   ├── test/  testXXXX.txt
│   └── labels_train.csv            # ReviewID, Rating (1–10)
└── submission.csv                  # Output predictions
```

> **Note**  
> You never commit `data/`. Any fresh clone only needs `dev-acad-nlp.zip`; the notebook unzips and processes it automatically.

---

## 🏗️ Model Architecture

```
Input (raw text)
  → TextVectorization (30k vocab, 250 seq len)
  → Embedding (128 dim)
  → LSTM (64 units)
  → Dropout 0.5
  → Dense (32, ReLU)
  → Dropout 0.3
  → Dense (10, Softmax) → Rating (1–10)
```

* **Loss:** `sparse_categorical_crossentropy`
* **Optimiser:** `Adam`
* **Metrics:** Accuracy

---

## 🚀 Quick-start (WSL 2 + Conda + GPU)

```bash
# Clone repo
git clone https://github.com/YOUR-ORG/Review-Rating-LSTM.git
cd Review-Rating-LSTM

# Create environment
conda create -n review-lstm python=3.11 -y
conda activate review-lstm

# Install requirements
pip install -r requirements.txt
pip install --upgrade tensorflow==2.19.*
python -m spacy download en_core_web_sm

# (Optional) place dev-acad-nlp.zip in root
# Run notebook
jupyter lab LSTM_Text_Classification.ipynb
```

### CUDA note

TensorFlow 2.19 for WSL includes its own CUDA runtime; no toolkit installation is needed inside WSL. Make sure your Windows NVIDIA driver is version 535+.

The notebook will:

1. Unzip *dev-acad-nlp* → `data/`
2. Clean & lemmatise reviews
3. Vectorise & pad sequences
4. Train/test the LSTM model
5. Save `submission.csv`

---

## 📊 Reproducing Results

With default hyperparameters, the notebook reaches **~0.48 macro accuracy** on the validation set. You may experiment with:

* `MAX_NUM_WORDS` (vocab size)
* `EMBEDDING_DIM`
* LSTM units
* Dropout rates
* Batch size
* Learning rate scheduler

---

## 📝 Requirements

* Python ≥ 3.11
* **TensorFlow 2.19 (GPU)**
* spaCy ≥ 3.8
* scikit-learn, pandas, numpy, matplotlib, seaborn

See `requirements.txt` for details.

---

## 🤝 Contributing

Pull requests welcome! Please open an issue to discuss major changes.

1. Fork & create a branch `feat/my-feature`
2. Commit with clear messages
3. Verify notebook & outputs
4. Submit a PR 🎉

---
