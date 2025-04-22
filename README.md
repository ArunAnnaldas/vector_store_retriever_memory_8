# vector_store_retriever_memory_8

# Semantic Memory Agent with FAISS (LangChain + OpenAI)

This project demonstrates how to build a **context-aware conversational agent** using [LangChain](https://www.langchain.com/), OpenAI GPT models, and FAISS. It leverages `VectorStoreRetrieverMemory` to create **semantic memory** — allowing the agent to retrieve and reuse relevant knowledge from prior conversations even when the input phrasing changes.

---

## VectorStoreRetrieverMemory

### 🎯 Goal:
Build a semantic-memory-powered assistant that:
- Understands user context across multiple prompts
- Recalls past topics based on meaning (not exact keywords)
- Uses FAISS for fast and efficient vector-based retrieval

---

## 🧠 Features

- Semantic memory storage using FAISS and OpenAI Embeddings
- Auto-retrieves top-k relevant memory for each new input
- Demonstrates meaningful memory chaining over conversation
- Tracks token usage via OpenAI callback
- Compatible with LangChain 0.2+ (fully updated imports)

---

## 💻 Example Interaction

```text
User: I need a function to convert Celsius to Fahrenheit
AI: Sure! F = C * 9/5 + 32 ...

User: Now add Kelvin support to the function we discussed
AI: Great idea! To convert Kelvin to Fahrenheit...

---

## 🛠 Tech Stack
Python 3.9+

LangChain (≥ 0.2.0)

OpenAI GPT-3.5

FAISS for vector similarity search

OpenAI Embeddings

VectorStoreRetrieverMemory
