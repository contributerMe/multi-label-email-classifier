# 📨 Multi-Label Email Classifier using fine tuned LLaMA 3.2 (1B)

This project demonstrates how to build a **multi-label email classification system** using **Meta LLaMa 3.2 (1B)** model, fine-tuned using **Parameter-Efficient Fine-Tuning (PEFT)** and deployed using **Gradio on Hugging Face Spaces**. The model can be accessed with [Multi_label Email Classifier](https://huggingface.co/spaces/imnim/Multi-labelEmailClassifier)   

---

##  Overview

Given an email's **subject** and **body**, the model predicts one or more relevant **categories** from a predefined set.

---

##  Classification Categories

The model classifies emails into the following 10 categories:

```python
CATEGORIES = [
    "Business", "Personal", "Promotions", "Customer Support", "Job Application",
    "Finance & Bills", "Events & Invitations", "Travel & Bookings",
    "Reminders", "Newsletters"
]
```

--- 

## 📊 Dataset Summary

We prepared a synthetic but realistic dataset of 2,105 labeled emails. Each email includes a subject, body, and one or more categories.

- **Total Emails**: `2105`
- **Label Distribution**:

```
Business               : 941
Customer Support       : 227
Events & Invitations   : 656
Finance & Bills        : 334
Job Application        : 122
Newsletters            : 199
Personal               : 221
Promotions             : 120
Reminders              : 343
Travel & Bookings      : 314
```

The dataset can be accessed with [Multi-label mail_dataset](https://huggingface.co/datasets/imnim/multiclass-email-classification)

---

##  Model Training

We fine-tuned [`meta-llama/Llama-3.1-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-1B-Instruct) using the [PEFT](https://github.com/huggingface/peft) library with LoRA adapters. Training was done with:

- 🤗 Hugging Face transformers
-  LoRA (for efficient fine-tuning)
-  Resulting adapter uploaded to Hugging Face Hub:
  
  `imnim/multi-label-email-classifier`
  [Model card on hugging face](https://huggingface.co/imnim/multi-label-email-classifier)

---


##  Backend & Deployment Options

### Local (FastAPI + React)

We initially set up a FastAPI backend and a React frontend, but:

- The model was too large to run on CPU.
- No GPU available locally.
- Hugging Face Spaces does **not support React + FastAPI apps with GPU** on free tier.

### Gradio on Hugging Face Spaces


Gradio apps run smoothly on Hugging Face's **free CPU** tier. Our `Gradio` version accepts subject and body inputs and returns predicted labels.


---

## 🌍 Access the App

➡️ [https://huggingface.co/spaces/imnim/Multi-labelEmailClassifier](https://huggingface.co/spaces/imnim/Multi-labelEmailClassifier)

---






