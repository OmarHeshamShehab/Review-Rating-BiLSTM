# Review-Rating-BiLSTM

A fully–reproducible deep-learning pipeline for predicting **1–10 star ratings** from free-form movie reviews.  
The project was developed as part of the *Dev-Acad NLP* challenge and demonstrates modern natural‑language‑processing (NLP) techniques with TensorFlow 2/Keras, spaCy and Scikit‑Learn.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Automatic data rebuild** | The `data/` folder is *not* version-controlled. If it is missing, the notebook zips the raw `dev-acad-nlp` corpus and recreates the same structure on the fly. |
| **Lightweight preprocessing** | Tokenisation, stop‑word removal & lemmatisation via *spaCy `en_core_web_sm`*. |
| **Deep Bi‑LSTM architecture** | Two stacked bidirectional LSTM layers capture long‑range context in both directions. |
| **Robust training utilities** | Early Stopping and Reduce‑LR‑on‑Plateau callbacks prevent over‑fitting. |
| **Single‑command submission** | Generates a ready‑to‑upload `submission.csv` that meets the competition format. |

---

## 🗂️ Project Structure

```text
.
├── LSTM_Text_Classification.ipynb   # End‑to‑end workflow
├── dev-acad-nlp.zip                # Raw corpus (11 k train + 3 k test)
├── data/                           # Auto‑generated working directory
│   ├── train/ trainXXXX.txt        # Cleaned review texts (Latin‑1)
│   ├── test/  testXXXX.txt
│   └── labels_train.csv            # ReviewID,Rating (1–10)
└── submission.csv                  # Sample or model output
```

> **Note**  
> You never commit `data/`. Any fresh clone only needs `dev-acad-nlp.zip`; the notebook will unzip it the first time it runs.

---

## 🏗️ Model Architecture

```
Input → Embedding (30 k vocab, 128 d)
      → Bi‑LSTM (64 units, return_sequences)
      → Bi‑LSTM (32 units) 
      → Dropout 0.5
      → Dense 32 ReLU
      → Dropout 0.3
      → Dense 10 Softmax → Rating (1‑10)
```

* **Loss** : `sparse_categorical_crossentropy`  
* **Optimiser** : `Adam`  
* **Metrics** : Accuracy

---

## 🚀 Quick‑start

```bash
# Clone repo
git clone https://github.com/YOUR-ORG/Review-Rating-BiLSTM.git
cd Review-Rating-BiLSTM

# Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# (Optional) place dev-acad-nlp.zip in project root
# Run the notebook
jupyter lab LSTM_Text_Classification.ipynb
```

The notebook will:

1. Unzip *dev-acad-nlp* ➜ `data/`  
2. Clean & lemmatise texts  
3. Train/validate the Bi‑LSTM (80 / 20 split)  
4. Predict ratings for the hidden test set  
5. Save `submission.csv`

---

## 📊 Reproducing the baseline score

With the default hyper‑parameters the notebook achieves **~0.48 macro accuracy** on the held‑out validation set (exact numbers may vary due to random seeds and GPU). Feel free to tweak:

* `MAX_NUM_WORDS` – vocabulary size  
* `EMBEDDING_DIM` – embedding dimension  
* LSTM units / layers  
* Dropout rates & batch size  
* Learning‑rate schedule

---

## 📝 Requirements

* Python ≥ 3.9  
* TensorFlow 2.15  
* spaCy ≥ 3.7  
* matplotlib, seaborn, scikit‑learn, pandas, numpy

A full `requirements.txt` is provided.

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

1. Fork the repo & create a branch `feat/my-awesome-improvement`  
2. Commit with conventional messages  
3. Ensure the unit tests/notebook still run  
4. Submit a PR 🎉

---

## 🪪 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

* **Dev-Acad** for releasing the dataset.  
* *spaCy* & *TensorFlow* teams for their brilliant libraries.  
* Inspired by myriad Kaggle kernels on text classification.

---

<div align="center">

_“Talk is cheap. Show me the data.”_  
— *Linus Torvalds*

</div>
