# NOVA Troubleshooting Guide

This guide documents common errors, environment constraints, and how to recover from them.

---

## 💾 1. PyTorch CUDA Out-Of-Memory / DLL Failures (WinError 1455)

### Symptoms
In Windows environments, running PyTorch CUDA operations concurrently can cause memory allocations to hang or throw:
`OSError: [WinError 1455] The paging file is too small for this operation to complete.`

### Fix
Force the application to execute in CPU mode by setting the `NOVA_FORCE_CPU` flag:
* **Windows Powershell**:
  ```powershell
  $env:NOVA_FORCE_CPU="true"
  ```
* **Linux/macOS**:
  ```bash
  export NOVA_FORCE_CPU="true"
  ```
Setting this flag tells both the vision classifier and SentenceTransformer embedding modules to ignore CUDA hooks and run cleanly on the CPU.

---

## 🗃️ 2. FAISS Index Errors (IndexFlatL2 Dimension Mismatches)

### Symptoms
`ValueError: dict dimensions or indexing shapes do not match 384.`

### Fix
NOVA uses the SentenceTransformer `sentence-transformers/all-MiniLM-L6-v2` which produces dense embeddings of size **384**. If you modify the settings configuration to a model with a different dimensionality (e.g., `all-mpnet-base-v2` with 768 dimensions), you must clear the existing FAISS index:
1. Delete the folder `data/faiss_index` or the index file.
2. Restart the backend. The startup script will initialize an empty index with the correct dimensions automatically.
