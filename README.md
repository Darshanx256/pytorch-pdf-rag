# A PDF-RAG ChatBot
  A from scratch made RAG Pipeline featuring custom PyTorch inference loop for manual token sampling.

# Architecture:
- PyMuPDF for processing PDFs (handling text extraction and sentence-aware chunking)
- FAISS for vector retrieval (using IndexFlatIP and All-MiniLM-L6-v2 embeddings)
- Qwen2.5-0.5B-Instruct for generation (loaded via HuggingFace AutoModelForCausalLM)

# The custom loop:
  Includes a custom loop to manually implement temperature scaling, softmax probability distribution  and multinomial sampling (using logits / tau, torch.softmax, and torch.multinomial). Also includes manual string slicing to clean up conversational text dumps.

ChatBot memory:
  Holds context of previous chats by injecting a memory_string of past Q&A tuples directly into the prompt tensor.

Future Improvements:
  Implementing Query Rewriting, using LLM to contextualize the query before FAISS retrival.
