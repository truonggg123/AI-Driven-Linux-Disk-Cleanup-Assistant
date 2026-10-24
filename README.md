# 🧹 AI-Driven Linux Disk Cleanup Assistant

An intelligent, safety-first disk cleanup assistant for Linux. It combines **Machine Learning (Decision Trees)** and **Natural Language Processing (NLP)** to analyze file semantics, size, and access age—recommending whether to **Delete**, **Compress**, or **Keep** files.

---

## ✨ Features

- **🤖 Smart NLP & AI Classification:** Analyzes file names using **spaCy** (`en_core_web_lg`) and **Sentence-Transformers** (`all-MiniLM-L6-v2`) embeddings reduced via **PCA** to classify files accurately.
- **📦 3 Action Recommendations:**
  - **Delete:** Flags temporary/cache files (`.log`, `.tmp`, `.cache`) or small files unused for > 180 days.
  - **Compress:** Flags large files (> 50MB) inactive for > 90 days and archives them into `.zip` to save space.
  - **Keep:** Protects active, recent, or critical files.
- **🛡️ Built-in Safety Guards:**
  - **System Path Blacklist:** Prevents scanning or modifying critical system directories (`/bin`, `/etc`, `/boot`, `/sys`, `~/.ssh`, etc.).
  - **Activity Buffer:** Skips files modified within the last 7 days.
  - **Safe Deletion:** Supports OS Recycle Bin via `send2trash` or local quarantine folder.
- **🖥️ Intuitive GUI:** Clean Tkinter desktop interface with checkboxes to review actions before execution.
- **🧪 Test Sandbox Included:** Easily generate dummy test files to test the model safely.

---

## 📁 Project Structure

```text
AI-Driven-Linux-Disk-Cleanup-Assistant/
├── disk_cleaner.py         # Main GUI application (Tkinter)
├── train_model.py          # Synthetic dataset generator & model trainer
├── create_test_folder.py   # Script to generate sandbox test files (ml_test_sandbox/)
├── install_pytorch_cpu.sh  # Script to install CPU-only PyTorch on Linux
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

Python **3.8+** is recommended. Create and activate a virtual environment:

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# 1. Install CPU-only PyTorch (recommended for lightweight CPU inference)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install required packages
pip install -r requirements.txt

# 3. Download the spaCy language model
python -m spacy download en_core_web_lg
```

> **Note:** The transformer model (`sentence-transformers/all-MiniLM-L6-v2`) downloads automatically on first run.

---

## 📖 How to Use

### Step 1: Train the Model (Required once)

Generate synthetic metadata and train the classifier:

```bash
python train_model.py
```
*This generates trained assets in `ml_assets/` (`disk_model.joblib`, `label_encoder.joblib`, `pca_transformer.joblib`).*

---

### Step 2: (Optional) Create Test Sandbox

Generate a safe test folder with 9 sample files (3 to Delete, 3 to Compress, 3 to Keep):

```bash
python create_test_folder.py
```

---

### Step 3: Launch the Disk Cleaner GUI

```bash
python disk_cleaner.py
```

1. Click **Browse...** ("Duyệt...") to select a directory (e.g. `ml_test_sandbox`).
2. Click **Analyze** ("Phân tích") to scan and predict recommended actions.
3. Check (`❌`) the files you want to process.
4. Toggle **Safe Delete** ("Xóa an toàn") as desired.
5. Click **Execute Actions** ("Thực hiện hành động") to clean or compress selected files.
6. Click **View History** ("Xem lịch sử") to review past cleanup actions.

---

## ⚙️ Decision Logic Summary

| Action | Criteria |
| :--- | :--- |
| **Delete** | Temporary/cache extensions (`.tmp`, `.log`, `.bak`, `.cache`), junk keywords in filename (`temp`, `cache`, `backup`), or small files (< 5MB) unused for > 180 days. |
| **Compress** | File size > 50MB, unused for > 90 days, not already compressed, and does not contain important keywords (`important`, `document`, `report`). |
| **Keep** | Important files, recently modified files (< 7 days), or files not matching delete/compress rules. |

---

## ⚠️ Safety Notice

- Always review the selected items before clicking **Execute Actions**.
- Keep **Safe Delete** enabled to send deleted files to your OS Trash/Recycle Bin rather than permanently deleting them.