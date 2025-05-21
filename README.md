# Review-Rating-BiLSTM

A fully–reproducible deep‑learning pipeline for predicting **1–10 star ratings** from free‑form movie reviews.  
The project was developed as part of the *Dev‑Acad NLP* challenge and demonstrates modern natural‑language‑processing (NLP) techniques with **TensorFlow 2.19 (GPU)**, spaCy and Scikit‑Learn. The instructions below assume you are running under **WSL 2** on Windows with an NVIDIA GPU.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Automatic data rebuild** | The `data/` folder is *not* version‑controlled. If it is missing, the notebook unzips the raw `dev‑acad‑nlp` corpus and recreates the same structure on the fly. |
| **Lightweight preprocessing** | Tokenisation, stop‑word removal & lemmatisation via *spaCy `en_core_web_sm`*. |
| **Deep Bi‑LSTM architecture** | Two stacked bidirectional LSTM layers capture long‑range context in both directions. |
| **Robust training utilities** | Early Stopping and Reduce‑LR‑on‑Plateau callbacks prevent over‑fitting. |
| **Single‑command submission** | Generates a ready‑to‑upload `submission.csv` that meets the competition format. |

---

## 🗂️ Project Structure

```text
.
├── LSTM_Text_Classification.ipynb   # End‑to‑end workflow
├── dev-acad-nlp.zip                # Raw corpus (11 k train + 3 k test)
├── data/                           # Auto‑generated working directory
│   ├── train/ trainXXXX.txt        # Cleaned review texts (Latin‑1)
│   ├── test/  testXXXX.txt
│   └── labels_train.csv            # ReviewID,Rating (1–10)
└── submission.csv                  # Sample or model output
```

> **Note**  
> You never commit `data/`. Any fresh clone only needs `dev‑acad‑nlp.zip`; the notebook will unzip it the first time it runs.

---

## 🏗️ Model Architecture

```
Input → Embedding (30 k vocab, 128 d)
      → Bi‑LSTM (64 units, return_sequences)
      → Bi‑LSTM (32 units) 
      → Dropout 0.5
      → Dense 32 ReLU
      → Dropout 0.3
      → Dense 10 Softmax → Rating (1–10)
```

* **Loss:** `sparse_categorical_crossentropy`
* **Optimiser:** `Adam`
* **Metrics:** Accuracy

---

## 🚀 Quick‑start (WSL 2 + Conda + GPU)

```bash
# Clone repo
git clone https://github.com/YOUR-ORG/Review-Rating-BiLSTM.git
cd Review-Rating-BiLSTM

# Create GPU‑enabled Conda environment with Python 3.11
conda create -n review-bilstm python=3.11 -y
conda activate review-bilstm

# Install core dependencies
pip install -r requirements.txt

# Install TensorFlow 2.19 GPU build (WSL automatically mounts CUDA)
pip install --upgrade tensorflow==2.19.*

# spaCy English language model
python -m spacy download en_core_web_sm

# (Optional) place dev-acad-nlp.zip in project root
# Run the notebook
jupyter lab LSTM_Text_Classification.ipynb
```

### CUDA note

TensorFlow 2.19 for WSL ships with its own CUDA runtime; you do **not** need a separate CUDA toolkit inside the distro. Ensure your Windows NVIDIA driver is 535+.

The notebook will:

1. Unzip *dev‑acad‑nlp* ➜ `data/`
2. Clean & lemmatise texts
3. Train/validate the Bi‑LSTM (80 / 20 split)
4. Predict ratings for the hidden test set
5. Save `submission.csv`

---

## 📊 Reproducing the baseline score

With the default hyper‑parameters the notebook achieves **~0.48 macro accuracy** on the held‑out validation set (exact numbers may vary). Feel free to tweak:

* `MAX_NUM_WORDS` – vocabulary size
* `EMBEDDING_DIM` – embedding dimension
* LSTM units / layers
* Dropout rates & batch size
* Learning‑rate schedule

---

## 📝 Requirements

* Python ≥ 3.10
* **TensorFlow 2.19 (GPU)**
* spaCy ≥ 3.7
* matplotlib, seaborn, scikit‑learn, pandas, numpy

A full `requirements.txt` is provided.

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

1. Fork the repo & create a branch `feat/my-awesome-improvement`
2. Commit with conventional messages
3. Ensure the unit tests/notebook still run
4. Submit a PR 🎉

---
