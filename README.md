# HealthSense_RAG

Summary Of The Project:
We developed a retrieval-augmented model where the user submits a query, and the system first searches its local storage to gather relevant context. If the available data is insufficient, the model automatically calls an external API to fetch additional information from PubMed. Using FAISS, it retrieves the top 10 most relevant articles based on the query. A reward model then evaluates these articles and selects the best three documents to use as final context.

To ensure reliable and interpretable outputs, the model follows a strict structured-response format. It first determines whether the user’s query represents pseudoscience or a scientifically supported fact. It then provides an answer grounded exclusively in the retrieved context, along with proper citations and the names of the articles used.

The base model is a 7-billion-parameter LLaMA model, which was further refined using Supervised Fine-Tuning (SFT) and GRPO to enforce structured responses and improve factual correctness.
