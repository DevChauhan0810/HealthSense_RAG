# HealthSense_RAG

🚀 *A Retrieval-Augmented System for Scientific, Context-Grounded Health Query Analysis*

---

## 📌 Project Summary

HealthSense_RAG is a Retrieval-Augmented Generation (RAG) system designed to deliver scientifically grounded, citation-backed, and structured responses to health-related queries.

The system workflow:

- Searches a **local FAISS vector store** for relevant context.
- If local context is insufficient, it automatically retrieves additional documents from **PubMed API**.
- FAISS retrieves the **top 10 most relevant articles**, which are then evaluated by a **reward model**.
- The reward model selects the **top 3 highest-quality documents** for final context.

The model produces strictly structured outputs:

- Classifies the query as **pseudoscience** or **scientific fact**.
- Generates an answer **only using retrieved context**.
- Includes **article names and citations** for all sources used.

The base LLM is a **7B-parameter LLaMA model**, further refined using:

- **Supervised Fine-Tuning (SFT)**  
- **GRPO (Reinforcement Optimization)**  

These steps ensure factual accuracy, consistency, and structure.

---

## 📥 Model Download Script

```python
from huggingface_hub import login
login(new_session=False)

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_DIR = "/kaggle/working/Llama-3.2-1B"

os.makedirs(MODEL_DIR, exist_ok=True)

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_ID,
    use_fast=True
)

# Download model
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    torch_dtype=torch.float16,
    device_map=None,
    low_cpu_mem_usage=True,
)

# Save to local directory
tokenizer.save_pretrained(MODEL_DIR)
model.save_pretrained(MODEL_DIR)

print("Model downloaded and stored at:", MODEL_DIR)
