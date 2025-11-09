import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import pandas as pd
import joblib
import time
import numpy as np
import zipfile
import os
from datetime import datetime
from send2trash import send2trash
import spacy
import gc

# Attempt to import PyTorch/Transformers, but ensure the app still runs if missing
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[Warning] PyTorch/Transformers not installed. Using spaCy only.")

# CONSTANTS
ASSETS_DIR = Path("ml_assets")
MODEL_FILE = ASSETS_DIR / "disk_model.joblib"
ENCODER_FILE = ASSETS_DIR / "label_encoder.joblib"
PCA_FILE = ASSETS_DIR / "pca_transformer.joblib" 
TEMP_EXTS = [".log", ".tmp", ".bak", ".cache", ".~", ""]
TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# FIXED DIRECTORY FOR CLEANER ARTIFACTS
CLEANER_ARTIFACTS_DIR = "ML_CLEANER_ARTIFACTS" 

# Get current user home directory
HOME_DIR = str(Path.home())

# BLACKLIST OF CRITICAL SYSTEM DIRECTORIES ON LINUX (Critical Paths)
CRITICAL_PATHS = [
    "/",
    "/bin", "/sbin", "/lib", "/lib64", "/usr/bin", "/usr/sbin", "/usr/lib", "/usr/lib64", 
    "/usr/local", "/opt", "/etc", "/boot", "/root", "/dev", "/proc", "/sys", "/run", 
    "/mnt", "/media", "/snap", "/lost+found", "/var/run", "/var/lock", "/var/log", 
    f"{HOME_DIR}/.ssh", f"{HOME_DIR}/.config", f"{HOME_DIR}/.local",
]

# ----------------- TRANSFORMERS EMBEDDING EXTRACTION FUNCTION -----------------
def get_transformer_embedding(text, tokenizer, model, max_length=128):
    if tokenizer is None or model is None or not HAS_TRANSFORMERS:
        return None
    
    try:
        import torch
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                              truncation=True, padding=True)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling to get the sentence embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        return embeddings.numpy()
    except Exception:
        return None

# ----------------- FEATURE EXTRACTION FUNCTION FROM FILENAME USING SPACY + TRANSFORMERS -----------------
def extract_spacy_features(file_path, nlp_model, transformer_tokenizer=None, transformer_model=None):
    file_name = Path(file_path).stem.lower()
    doc = nlp_model(file_name)
    
    num_words = len([token for token in doc if token.is_alpha])
    name_length = len(file_name)
    temp_keywords = ['temp', 'tmp', 'cache', 'old', 'backup', 'bak', '~']
    has_temp_keyword = any(keyword in file_name for keyword in temp_keywords)
    important_keywords = ['important', 'final', 'document', 'report']
    has_important_keyword = any(keyword in file_name for keyword in important_keywords)
    
    if len(doc) > 0 and doc.vector is not None:
        spacy_embedding = doc.vector 
    else:
        spacy_embedding = np.zeros(300) 
    
    transformer_embedding = get_transformer_embedding(file_name, transformer_tokenizer, transformer_model)
    if transformer_embedding is None or len(transformer_embedding) != 384:
        transformer_embedding = np.zeros(384) 
    
    combined_embedding = np.concatenate([spacy_embedding, transformer_embedding])
    
    return {
        'num_words': num_words,
        'name_length': name_length,
        'has_temp_keyword': int(has_temp_keyword),
        'has_important_keyword': int(has_important_keyword),
        'embedding_vector': combined_embedding
    }

class DiskCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Disk Cleaner Assistant") # Trợ lý Dọn Dẹp Đĩa ML
        self.root.geometry("900x600")

        # try-except block for model loading
        try:
            self.model = joblib.load(MODEL_FILE)
            self.encoder = joblib.load(ENCODER_FILE)
            self.pca = joblib.load(PCA_FILE)
        except FileNotFoundError as e:
            messagebox.showerror("Initialization Error", f"ML/PCA model files not found.\nDetails: {e}") # Lỗi Khởi tạo
            self.root.destroy()
            return
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_lg") 
        except OSError:
            messagebox.showerror("Initialization Error", "spaCy model 'en_core_web_lg' not found.\nPlease run: python -m spacy download en_core_web_lg") # Lỗi Khởi tạo
            self.root.destroy()
            return
        
        # Initialize Transformers model 
        self.transformer_tokenizer = None
        self.transformer_model = None
        if HAS_TRANSFORMERS:
            try:
                import torch
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
                self.transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME).to('cpu')
                self.transformer_model.eval()
            except Exception as e:
                print(f"[Warning] Could not load Transformers model: {e}") # Cảnh báo
        
        self.folder_path = tk.StringVar()
        self.history = []
        self.selected_folder_path = None  # Save the selected root folder

        self.create_widgets()

    def create_widgets(self):
        # --- Folder Selection ---
        path_frame = tk.Frame(self.root)
        path_frame.pack(pady=10)

        tk.Label(path_frame, text="Select Folder:").grid(row=0, column=0, padx=5) # Chọn thư mục
        tk.Entry(path_frame, textvariable=self.folder_path, width=60).grid(row=0, column=1)
        tk.Button(path_frame, text="Browse...", command=self.browse_folder).grid(row=0, column=2, padx=5) # Duyệt

        # --- Function Buttons Group ---
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Analyze", width=20, command=self.analyze).grid(row=0, column=0, padx=5) # Phân tích
        tk.Button(button_frame, text="Execute Actions", width=20, command=self.execute_actions).grid(row=0, column=1, padx=5) # Thực hiện hành động
        tk.Button(button_frame, text="View History", width=20, command=self.show_history).grid(row=0, column=2, padx=5) # Xem lịch sử
        tk.Button(button_frame, text="Select All", width=20, command=self.toggle_select_all).grid(row=0, column=3, padx=5) # Chọn tất cả

        # --- Progress bar and status label ---
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready") # Sẵn sàng
        self.status_label = tk.Label(progress_frame, textvariable=self.progress_var, anchor="w")
        self.status_label.pack(fill="x")
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.progress_bar.pack(fill="x", pady=5)

        # --- Treeview to display list ---
        tree_frame = tk.Frame(self.root)
        tree_frame.pack(expand=True, fill="both", padx=10, pady=10)

        scrollbar = tk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")

        self.tree = ttk.Treeview(tree_frame, columns=("Select", "Path", "Size", "Days", "Action"), show="headings", yscrollcommand=scrollbar.set)
        self.tree.column("Select", width=50, anchor="center") # Chọn
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
        self.tree.pack(expand=True, fill="both")
        scrollbar.config(command=self.tree.yview)
        self.tree.bind("<Button-1>", self.toggle_selection)

    def is_critical_path(self, folder_str):
        """Checks if the path is within CRITICAL_PATHS.""" # Kiểm tra xem đường dẫn có nằm trong CRITICAL_PATHS hay không
        folder_str = str(Path(folder_str)) 
        
        for cp in CRITICAL_PATHS:
            # Check for exact match OR starts with CP + "/" (is a subdirectory)
            if folder_str == cp or folder_str.startswith(cp + os.sep): 
                return True
        return False

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            # Safety Layer 1: Block critical directories immediately upon selection
            if self.is_critical_path(folder):
                messagebox.showerror(
                    "Permission Error", # Lỗi Quyền truy cập
                    "Permission denied."
                )
            else:
                self.folder_path.set(folder)
                self.selected_folder_path = folder  # Save the root folder

    def toggle_selection(self, event):
        """Toggle selection for the 'Select' column.""" # Chọn hoặc bỏ chọn cột 'Chọn'
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if item and col == "#1":
            values = list(self.tree.item(item, "values"))
            values[0] = "❌" if values[0] == "" else "" 
            self.tree.item(item, values=values)

    def toggle_select_all(self):
        """Select or deselect all items in the treeview""" # Chọn hoặc bỏ chọn tất cả các items trong treeview
        items = self.tree.get_children()
        if not items:
            return
        
        # Check if all are already selected
        all_selected = all(self.tree.item(item)['values'][0] == "❌" for item in items)
        
        # If all are selected, deselect all. Otherwise, select all
        new_value = "" if all_selected else "❌"
        
        for item in items:
            values = list(self.tree.item(item, "values"))
            values[0] = new_value
            self.tree.item(item, values=values)

    def analyze(self):
        folder_str = self.folder_path.get()
        folder = Path(folder_str)

        if not folder.is_dir():
            messagebox.showerror("Error", "Invalid folder.") # Lỗi, Thư mục không hợp lệ
            return

        # Safety Layer 2: Block critical directories if user manually enters
        if self.is_critical_path(folder_str):
            messagebox.showerror(
                "Permission Error", # Lỗi Quyền truy cập
                "Permission denied."
            )
            return

        try:
            # Initialize progress bar
            self.progress_bar['value'] = 0
            self.progress_var.set("Scanning folder...") # Đang quét thư mục
            self.root.update()
            
            self.df, error_msg = self.scan_folder(folder) # Receive error_msg too
            
            # CHECK FOR PERMISSION ERROR FROM SCANNER (Only display "Permission denied.")
            if error_msg:
                 messagebox.showerror("Permission Error", "Permission denied.") # Lỗi Quyền truy cập
                 self.tree.delete(*self.tree.get_children())
                 self.progress_bar['value'] = 0
                 self.progress_var.set("Ready") # Sẵn sàng
                 return
                 
            if self.df.empty:
                messagebox.showinfo("Notice", "No eligible files found.") # Thông báo, Không có tệp nào đủ điều kiện
                self.tree.delete(*self.tree.get_children())
                self.progress_bar['value'] = 0
                self.progress_var.set("Ready") # Sẵn sàng
                return

            # Update progress: 20% after scan
            self.progress_bar['value'] = 20
            self.progress_var.set(f"Scanned {len(self.df)} files. Calculating features...") # Đã quét ... tệp. Đang tính toán features
            self.root.update()

            self.df = self.calculate_features(self.df)
            
            feature_cols = ['size_log', 'days_since_access', 'is_temp_file',
                             'num_words', 'name_length', 'has_temp_keyword', 'has_important_keyword']
            embedding_cols = [col for col in self.df.columns if col.startswith('embedding_dim_')]
            feature_cols.extend(embedding_cols)
            
            available_cols = [col for col in feature_cols if col in self.df.columns]
            X = self.df[available_cols]
            
            # Update progress: 70% after feature calculation
            self.progress_bar['value'] = 70
            self.progress_var.set("Predicting with ML model...") # Đang dự đoán bằng mô hình ML
            self.root.update()
            
            # Handle prediction in batches if data is too large
            if len(X) > 2000:
                print("    Predicting in batches...") # Đang dự đoán theo batch
                preds_list = []
                pred_batch_size = 2000
                total_batches = (len(X) + pred_batch_size - 1) // pred_batch_size
                
                for batch_idx, pred_start in enumerate(range(0, len(X), pred_batch_size)):
                    pred_end = min(pred_start + pred_batch_size, len(X))
                    batch_X = X.iloc[pred_start:pred_end]
                    batch_preds = self.model.predict(batch_X)
                    preds_list.extend(batch_preds)
                    
                    # Update progress during prediction
                    progress_value = 70 + int((batch_idx + 1) / total_batches * 20)
                    self.progress_bar['value'] = progress_value
                    self.progress_var.set(f"Predicting: {pred_end}/{len(X)} files...") # Đang dự đoán
                    self.root.update()
                    
                    # Release memory
                    del batch_X, batch_preds
                    gc.collect()
                    
                preds = np.array(preds_list)
                del preds_list
            else:
                preds = self.model.predict(X)
                self.progress_bar['value'] = 90
                self.root.update()
            
            self.df['Predicted_Label'] = self.encoder.inverse_transform(preds)
            self.df['Formatted_Size'] = self.df['size_bytes'].apply(self.format_size)

            # Save root folder to calculate relative path
            self.selected_folder_path = folder_str

            # Update progress: 90% - displaying results
            self.progress_bar['value'] = 90
            self.progress_var.set("Displaying results...") # Đang hiển thị kết quả
            self.root.update()

            self.tree.delete(*self.tree.get_children())
            results_df = self.df[self.df['Predicted_Label'] != 'Keep']
            total_results = len(results_df)
            
            for idx, (_, row) in enumerate(results_df.iterrows()):
                full_path = row['file_path']
                # Calculate relative path compared to the root folder
                try:
                    relative_path = os.path.relpath(full_path, folder_str)
                except ValueError:
                    # If relative path cannot be calculated (different drive on Windows), use filename
                    relative_path = Path(full_path).name
                
                # Save full path to tag, display relative path
                self.tree.insert("", "end", values=("", relative_path, row['Formatted_Size'], f"{int(row['days_since_access'])} days", row['Predicted_Label']), tags=(full_path,)) # ngày
                
                # Update progress while displaying (every 1000 items)
                if (idx + 1) % 1000 == 0 or (idx + 1) == total_results:
                    progress_value = 90 + int((idx + 1) / total_results * 10)
                    self.progress_bar['value'] = progress_value
                    self.progress_var.set(f"Displaying: {idx + 1}/{total_results} results...") # Đang hiển thị ... kết quả
                    self.root.update()
            
            # Release memory
            del X, preds
            gc.collect()
            
            # Completion
            self.progress_bar['value'] = 100
            self.progress_var.set(f"Complete! Found {total_results} files to process.") # Hoàn thành! Tìm thấy ... tệp cần xử lý
            self.root.update()
            
            if not self.tree.get_children():
                 messagebox.showinfo("Notice", "The model suggested no actions for the scanned files.") # Thông báo, Mô hình không đề xuất hành động nào cho các tệp đã quét
                 self.progress_bar['value'] = 0
                 self.progress_var.set("Ready") # Sẵn sàng
                 
        except KeyboardInterrupt:
            messagebox.showwarning("Warning", "Analysis process stopped.") # Cảnh báo, Quá trình phân tích đã bị dừng
            self.tree.delete(*self.tree.get_children())
            self.progress_bar['value'] = 0
            self.progress_var.set("Ready") # Sẵn sàng
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"An error occurred during analysis:\n{error_msg}") # Lỗi, Đã xảy ra lỗi khi phân tích
            self.tree.delete(*self.tree.get_children())
            self.progress_bar['value'] = 0
            self.progress_var.set("Ready") # Sẵn sàng


    def scan_folder(self, folder):
        now = time.time()
        data = []
        folder_str = str(folder)
        
        # CRITICAL_PATHS safety layer 
        if self.is_critical_path(folder_str):
            return pd.DataFrame(), "Permission denied." # Return error message for critical path
            
        try:
            # CHECK ACCESS PERMISSION FOR THE ROOT FOLDER
            list(folder.iterdir()) 
        except PermissionError:
            # Catch error at the highest level and return a concise error message
            return pd.DataFrame(), "Permission denied."
        except Exception:
            return pd.DataFrame(), None 

        for path in folder.rglob("*"):
            path_str = str(path)
            
            # INSERT CHILD FILE FILTER SAFETY LAYER
            if self.is_critical_path(path_str):
                continue
                
            if path.is_file():
                try:
                    stat = path.stat()
                        
                    days = (now - stat.st_mtime) / (24 * 3600)
                    
                    # Filter: Only process files not accessed in the last 7 days
                    if days < 7:
                        continue
                        
                    ext = path.suffix.lower()
                    data.append({
                        "file_path": path_str,
                        "size_bytes": stat.st_size,
                        "extension": ext,
                        "days_since_access": days,
                        "is_temp_file": int(ext in TEMP_EXTS),
                        "size_log": np.log10(stat.st_size + 1)
                    })
                except PermissionError:
                    # If PermissionError occurs inside, we skip that file
                    continue
                except Exception:
                    continue
                    
        return pd.DataFrame(data), None # Return DataFrame and None (no error)

    def calculate_features(self, df):
        if df.empty: return df

        df['is_temp_file'] = df['extension'].isin(TEMP_EXTS).astype(int)
        df['size_log'] = np.log10(df['size_bytes'] + 1)
        
        print("    Extracting features from filename using spaCy + Transformers...") # Đang trích xuất features từ tên file bằng spaCy + Transformers
        
        # Process in batches to avoid memory overflow
        BATCH_SIZE = 500  # Reduce batch size to prevent memory overflow
        spacy_features = []
        embedding_vectors = []
        
        total_files = len(df)
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
        progress_start = 20  # Start at 20%
        progress_range = 50  # Occupy 50% of the progress bar (20% -> 70%)
        
        try:
            for batch_idx, batch_start in enumerate(range(0, total_files, BATCH_SIZE)):
                batch_end = min(batch_start + BATCH_SIZE, total_files)
                batch_df = df.iloc[batch_start:batch_end]
                
                batch_spacy_features = []
                batch_embedding_vectors = []
                
                for idx, file_path in enumerate(batch_df['file_path']):
                    try:
                        features = extract_spacy_features(file_path, self.nlp, self.transformer_tokenizer, self.transformer_model)
                        batch_spacy_features.append({
                            'num_words': features['num_words'],
                            'name_length': features['name_length'],
                            'has_temp_keyword': features['has_temp_keyword'],
                            'has_important_keyword': features['has_important_keyword']
                        })
                        batch_embedding_vectors.append(features['embedding_vector'])
                    except Exception as e:
                        # If an error occurs while processing a file, use default values
                        print(f"    Warning: Error processing {file_path}: {e}") # Cảnh báo: Lỗi khi xử lý
                        batch_spacy_features.append({
                            'num_words': 0,
                            'name_length': len(Path(file_path).stem),
                            'has_temp_keyword': 0,
                            'has_important_keyword': 0
                        })
                        # Create default embedding vector (300 spaCy + 384 Transformers = 684)
                        batch_embedding_vectors.append(np.zeros(684))
                
                spacy_features.extend(batch_spacy_features)
                embedding_vectors.extend(batch_embedding_vectors)
                
                # Update progress bar
                progress_value = progress_start + int((batch_idx + 1) / total_batches * progress_range * 0.8)  # 80% for feature extraction
                self.progress_bar['value'] = progress_value
                self.progress_var.set(f"Extracting features: {batch_end}/{total_files} files...") # Đang trích xuất features
                self.root.update()
                
                # Release memory after each batch
                del batch_spacy_features, batch_embedding_vectors, batch_df
                gc.collect()
                
                # Clear PyTorch cache if available to free up GPU/RAM memory
                if HAS_TRANSFORMERS:
                    try:
                        import torch
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    except:
                        pass
                
                if batch_end % 1000 == 0 or batch_end == total_files:
                    print(f"    Processed {batch_end}/{total_files} files...") # Đã xử lý ... tệp
            
            # Convert to DataFrame and process PCA in batches
            spacy_df = pd.DataFrame(spacy_features)
            df = pd.concat([df, spacy_df], axis=1)
            
            # Update progress: 60% before PCA
            progress_pca_start = progress_start + int(progress_range * 0.8)
            self.progress_bar['value'] = progress_pca_start
            self.progress_var.set("Reducing embedding dimensions with PCA...") # Đang giảm chiều embedding bằng PCA
            self.root.update()
            
            # Process PCA in batches to avoid memory overflow
            print("    Reducing embedding dimensions with PCA...") # Đang giảm chiều embedding bằng PCA
            embedding_matrix = np.array(embedding_vectors, dtype=np.float32)  # Use float32 to save memory
            
            # Process PCA in batches if matrix is too large
            if len(embedding_matrix) > 2000:
                # Split into smaller batches for PCA
                pca_batch_size = 2000
                pca_results = []
                pca_total_batches = (len(embedding_matrix) + pca_batch_size - 1) // pca_batch_size
                pca_progress_range = progress_range * 0.2  # Remaining 20% for PCA
                
                for pca_idx, pca_start in enumerate(range(0, len(embedding_matrix), pca_batch_size)):
                    pca_end = min(pca_start + pca_batch_size, len(embedding_matrix))
                    batch_matrix = embedding_matrix[pca_start:pca_end]
                    batch_reduced = self.pca.transform(batch_matrix)
                    pca_results.append(batch_reduced)
                    
                    # Update progress during PCA
                    progress_value = progress_pca_start + int((pca_idx + 1) / pca_total_batches * pca_progress_range)
                    self.progress_bar['value'] = progress_value
                    self.progress_var.set(f"Processing PCA: {pca_end}/{len(embedding_matrix)} files...") # Đang xử lý PCA
                    self.root.update()
                    
                    # Release memory
                    del batch_matrix, batch_reduced
                    gc.collect()
                
                embedding_reduced = np.vstack(pca_results)
                del pca_results
            else:
                embedding_reduced = self.pca.transform(embedding_matrix)
                self.progress_bar['value'] = progress_start + progress_range
                self.root.update()
            
            # Final memory release
            del embedding_matrix, embedding_vectors, spacy_features
            gc.collect()
            
            embedding_cols = [f'embedding_dim_{i}' for i in range(embedding_reduced.shape[1])]
            embedding_df = pd.DataFrame(embedding_reduced, columns=embedding_cols, index=df.index)
            
            df = pd.concat([df, embedding_df], axis=1)
            
            # Final memory release
            del embedding_reduced, embedding_df
            gc.collect()
            
            return df
            
        except KeyboardInterrupt:
            print("\n    Processing stopped by user request.") # Đã dừng xử lý do người dùng yêu cầu
            raise
        except Exception as e:
            print(f"    Error calculating features: {e}") # Lỗi khi tính toán features
            raise

    def format_size(self, size):
        if size >= 1024**3:
            return f"{size / 1024**3:.2f} GB"
        elif size >= 1024**2:
            return f"{size / 1024**2:.2f} MB"
        else:
            return f"{size / 1024:.2f} KB"

    def execute_actions(self):
        # PERMISSION ERROR FIX: Fix the archive directory to the home directory
        base_artifact_dir = Path.home() / CLEANER_ARTIFACTS_DIR
        
        try:
            base_artifact_dir.mkdir(exist_ok=True)
        except Exception as e:
            messagebox.showerror("Write Error", f"Cannot create archive directory at {base_artifact_dir}. Details: {e}") # Lỗi Ghi
            return
            
        archive_dir = base_artifact_dir / "ARCHIVE_ML_ASSISTANT"
        archive_dir.mkdir(exist_ok=True)
        
        # Count selected files first
        selected_items = []
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            if values[0] == "❌":
                selected_items.append(item)
        
        if not selected_items:
            messagebox.showinfo("Notice", "No files selected.") # Thông báo, Không có tệp nào được chọn
            return
        
        # Initialize progress bar
        self.progress_bar['value'] = 0
        self.progress_var.set(f"Processing {len(selected_items)} files...") # Đang xử lý
        self.root.update()
        
        count = 0
        has_permission_error = False # Flag to check for permission errors
        items_to_remove = []  # List of items to remove from treeview
        total_items = len(selected_items)
        
        # Iterate over all selected items
        for item_idx, item in enumerate(selected_items):
            values = self.tree.item(item)['values']
                
            # Get the full path from the tag (if available), otherwise use path from values
            tags = self.tree.item(item)['tags']
            if tags and len(tags) > 0:
                path = tags[0]  # Full path is stored in the tag
            else:
                # Fallback: if no tag, try to join with root folder
                path_from_values = values[1]
                if self.selected_folder_path:
                    path = str(Path(self.selected_folder_path) / path_from_values)
                else:
                    path = path_from_values
            
            _, _, _, action = values[1], values[2], values[3], values[4]
            file_path = Path(path)
            
            # FINAL SAFETY LAYER (CRITICAL PATHS)
            if self.is_critical_path(str(file_path)):
                print(f"Protection: Skipping action on critical system file: {file_path}")
                continue 

            # Update progress bar
            progress_value = int((item_idx + 1) / total_items * 100)
            self.progress_bar['value'] = progress_value
            self.progress_var.set(f"Processing: {item_idx + 1}/{total_items} files ({action})...") # Đang xử lý
            self.root.update()
            
            try:
                success = False
                if action == "Delete":
                    send2trash(file_path)
                    status = "Deleted (Trash)" # Xóa (Trash)
                    success = True
                    
                elif action == "Compress":
                    zip_path = archive_dir / f"{file_path.stem}.zip"
                    # If the user does not have write/delete permissions, the exception will occur here
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(file_path, file_path.name)  
                    file_path.unlink()
                    status = "Compressed and original deleted" # Nén và xóa gốc
                    success = True
                
                if success:
                    self.history.append({
                        "file": str(file_path),
                        "action": action,
                        "status": status,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    count += 1
                    # Mark this item for removal from treeview
                    items_to_remove.append(item)
                    
            except PermissionError as e:
                # CATCH PERMISSION ERROR AND SET FLAG
                has_permission_error = True
            except Exception as e:
                # Catch other I/O errors
                print(f"Error processing {file_path.name}: {e}")
        
        # Remove successfully processed items from treeview
        for item in items_to_remove:
            self.tree.delete(item)
        
        # Update progress bar completion
        self.progress_bar['value'] = 100
        self.progress_var.set(f"Complete! Executed {count}/{total_items} actions.") # Hoàn thành! Đã xử lý
        self.root.update()
                
        # --- DISPLAY FINAL RESULT ---
        if has_permission_error:
            # Display the most concise "Permission denied" message
            messagebox.showerror(
                "Permission Error", # Lỗi Quyền truy cập
                "Permission denied."
            )
            self.progress_bar['value'] = 0
            self.progress_var.set("Ready") # Sẵn sàng
        elif count == 0:
             # Notice when no files were selected or successfully processed
             messagebox.showinfo("Complete", "Executed 0 actions. Please check if files were selected.") # Hoàn tất, Đã thực hiện 0 hành động. Vui lòng kiểm tra đã chọn tệp chưa.
             self.progress_bar['value'] = 0
             self.progress_var.set("Ready") # Sẵn sàng
        else:
             # Update interface after completion (no need to re-run analysis)
             messagebox.showinfo("Complete", f"Executed {count} actions successfully.") # Hoàn tất, Đã thực hiện ... hành động thành công
             # Keep progress at 100% for a moment, then reset
             def reset_progress():
                 self.progress_bar['value'] = 0
                 self.progress_var.set("Ready") # Sẵn sàng
             self.root.after(2000, reset_progress)
             
    def show_history(self):
        if not self.history:
            messagebox.showinfo("History", "No actions recorded yet.") # Lịch sử, Chưa có hành động nào
            return
        hist_win = tk.Toplevel(self.root)
        hist_win.title("Action History") # Lịch sử hành động
        # Add Scrollbar to history window
        scrollbar = tk.Scrollbar(hist_win)
        scrollbar.pack(side="right", fill="y")
        tree = ttk.Treeview(hist_win, columns=("File", "Action", "Status", "Time"), show="headings", yscrollcommand=scrollbar.set)
        tree.column("File", width=400)
        tree.column("Action", width=150, anchor="center")
        tree.column("Status", width=250)
        tree.column("Time", width=150)
        for col in tree["columns"]:
            tree.heading(col, text=col)
        tree.pack(expand=True, fill="both")
        scrollbar.config(command=tree.yview)
        for row in self.history:
            tree.insert("", "end", values=(row["file"], row["action"], row["status"], row["time"]))

# RUN APPLICATION
if __name__ == "__main__":
    # Ensure assets directory exists (for model files)
    Path(ASSETS_DIR).mkdir(exist_ok=True) 
    
    root = tk.Tk()
    app = DiskCleanerApp(root)
    root.mainloop()
