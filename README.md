
---

# Supplementary Software for Hundred-layer photonic deep learning

This repository contains the supplementary software package accompanying the manuscript:
**Hundred-layer photonic deep learning**


## Contents

* Source code for algorithm implementation
* Example datasets
* Evaluation scripts

---

## 1. System Requirements

* **Operating System**: Ubuntu 5.15.0-136-generic
* **Python Version**: 3.9
* **Dependencies**: Please view requirements.txt

To simplify setup, we recommend using a virtual environment.

---

## 2. Installation Guide

1. **Set Up Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Instructions to Run

1. Navigate to the main directory(GPT-2):

   ```bash
   cd ./SentenceGen/gpt2
   ```

2. Setting up the training corpus:

   ```bash
   mkdir build
   ```
   Put the `corpus.train.txt`, `corpus.test.txt`, and `vocab.txt` into the `build` directory, which can be generated from the wiki dumps 20240220 or any text file you like.

3. Training stage instructions:

   ```bash
   python -m gpt2 train --train_corpus           build/corpus.train.txt \
                        --eval_corpus            build/corpus.test.txt \
                        --vocab_path             build/vocab.txt \
                        --save_model_path        gpt2-pretrained.pth \
                        --batch_train            16 \
                        --batch_eval             16 \
                        --total_steps            1000000 \
                        --eval_steps             100 \
                        --save_steps             500 \
                        --seq_len                64 \
                        --layers                 24 \
                        --dims                   1024 \
                        --heads                  16 \
                        --base_lr                1e-4
   ```

4. Testing stage:

   ```bash
   python -m gpt2 evaluate --model_path             gpt2-pretrained.pth \
                           --eval_corpus            build/corpus.evaluate.txt \
                           --vocab_path             build/vocab.txt \
                           --batch_eval             1 \
                           --total_steps            10 \
                           --seq_len                64 \
                           --layers                 24 \
                           --dims                   1024 \
                           --heads                  16 \
                           --use_gpu \
   ```

---

