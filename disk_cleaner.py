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
import gc
import threading
import queue
import math # Import math for log

# CONSTANTS
ASSETS_DIR = Path("ml_assets")
MODEL_FILE = ASSETS_DIR / "disk_model.joblib"
ENCODER_FILE = ASSETS_DIR / "label_encoder.joblib"
PCA_FILE = ASSETS_DIR / "pca_transformer.joblib" 
TEMP_EXTS = [".log", ".tmp", ".bak", ".cache", ".~", ""]
TEMP_EXTS_SET = set(TEMP_EXTS) # Optimization: Use set for faster checks

# FIXED DIRECTORY FOR CLEANUP ARTIFACTS
CLEANER_ARTIFACTS_DIR = "ML_CLEANER_ARTIFACTS" 

# Get current user's home directory
HOME_DIR = str(Path.home())

# BLACKLIST OF CRITICAL SYSTEM PATHS (use set for faster lookup)
CRITICAL_PATHS_SET = {
    "/", "/bin", "/sbin", "/lib", "/lib64", "/usr/bin", "/usr/sbin", 
    "/usr/lib", "/usr/lib64", "/usr/local", "/opt", "/etc", "/boot", 
    "/root", "/dev", "/proc", "/sys", "/run", "/mnt", "/media", "/snap", 
    "/lost+found", "/var/run", "/var/lock", "/var/log",
    f"{HOME_DIR}/.ssh", f"{HOME_DIR}/.config", f"{HOME_DIR}/.local"
}

# Keyword sets for faster lookup
TEMP_KEYWORDS = {'temp', 'tmp', 'cache', 'old', 'backup', 'bak', '~'}
IMPORTANT_KEYWORDS = {'important', 'final', 'document', 'report'}

# ----------------- SIMPLE FEATURE EXTRACTION FROM FILE NAME (MODEL-FREE) -----------------
def extract_simple_features(file_path):
    """Simplified, model-free feature extraction for performance"""
    file_name = Path(file_path).stem.lower()
    name_length = len(file_name)
    
    # Simple word count (split by non-alphanumeric)
    words = [w for w in file_name.split('_') + file_name.split('-') if w.isalnum()]
    num_words = len(words)
    
    # Keyword checks
    has_temp_keyword = int(any(kw in file_name for kw in TEMP_KEYWORDS))
    has_important_keyword = int(any(kw in file_name for kw in IMPORTANT_KEYWORDS))
    
    # Use zero vector of size 684 for compatibility with trained PCA
    spacy_embedding = np.zeros(684, dtype=np.float32)
    
    return {
        'num_words': num_words,
        'name_length': name_length,
        'has_temp_keyword': has_temp_keyword,
        'has_important_keyword': has_important_keyword,
        'embedding_vector': spacy_embedding
    }

class DiskCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Disk Cleanup Assistant")
        self.root.geometry("1200x700")

        # Try-except block for loading the model
        try:
            self.model = joblib.load(MODEL_FILE)
            self.encoder = joblib.load(ENCODER_FILE)
            self.pca = joblib.load(PCA_FILE)
        except FileNotFoundError as e:
            messagebox.showerror("Initialization Error", f"ML/PCA model files not found.\nDetails: {e}")
            self.root.destroy()
            return
        
        # Spacy is not used for performance optimization
        self.nlp = None

        self.folder_path = tk.StringVar()
        self.history = []
        
        # Queue for communication between background thread and main thread
        self.result_queue = queue.Queue()
        self.is_processing = False
        
        # Store references to buttons for enabling/disabling
        self.analyze_btn = None
        self.execute_btn = None
        self.history_btn = None
        self.browse_btn = None
        self.title_label = None
        
        # Store analysis results
        self.analysis_results_df = None

        self.create_widgets()
        
        # Check queue periodically to update GUI
        self.check_queue()
    
    # ... (other methods of DiskCleanerApp) ...
    
    def create_widgets(self):
        # --- Title ---
        self.title_label = ttk.Label(self.root, text="ML Disk Cleanup Assistant", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=(10, 15))
        
        # --- Select Folder ---
        path_frame = ttk.Frame(self.root)
        path_frame.pack(pady=10, padx=20, fill="x")
        
        path_label = ttk.Label(path_frame, text="Select Folder:")
        path_label.grid(row=0, column=0, padx=10, pady=10)
        
        path_entry = ttk.Entry(path_frame, textvariable=self.folder_path, width=60)
        path_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        path_frame.grid_columnconfigure(1, weight=1)
        
        self.browse_btn = ttk.Button(path_frame, text="Browse", command=self.browse_folder, width=10)
        self.browse_btn.grid(row=0, column=2, padx=10, pady=10)
        
        # --- Status and Progress Bar ---
        status_frame = ttk.Frame(self.root)
        status_frame.pack(pady=5, padx=20, fill="x")
        
        self.status_label = ttk.Label(status_frame, text="Ready", anchor="w")
        self.status_label.pack(side="left", padx=10, pady=8, fill="x", expand=True) # Fill and expand to take space
        
        self.progress_bar = ttk.Progressbar(status_frame, length=250, mode='determinate', maximum=1.0)
        self.progress_bar.pack(side="right", padx=10, pady=8)
        self.progress_bar['value'] = 0
        self.progress_bar.pack_forget()
        
        # --- Function Buttons ---
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10, padx=20, fill="x")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze", command=self.start_analyze)
        self.analyze_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.execute_btn = ttk.Button(button_frame, text="Execute", command=self.start_execute_actions, state="disabled") # Disabled by default
        self.execute_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        self.history_btn = ttk.Button(button_frame, text="History", command=self.show_history)
        self.history_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")
        
        # --- Treeview Display List ---
        self.summary_label = ttk.Label(self.root, text="")
        self.summary_label.pack(pady=(5, 5))
        
        tree_frame = ttk.Frame(self.root)
        tree_frame.pack(expand=True, fill="both", padx=20, pady=(0, 15))
        
        # Custom Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.tree = ttk.Treeview(
            tree_frame, 
            columns=("Select", "Path", "Size", "Days", "Action"), 
            show="headings", 
            yscrollcommand=scrollbar.set
        )
        self.tree.column("Select", width=60, anchor="center", stretch=tk.NO) 
        self.tree.column("Path", width=600, stretch=tk.YES)
        self.tree.column("Size", width=100, anchor="center", stretch=tk.NO)
        self.tree.column("Days", width=120, anchor="center", stretch=tk.NO)
        self.tree.column("Action", width=150, anchor="center", stretch=tk.NO)
        
        # Set column headings (already mostly English, but ensuring "Chọn" is "Select")
        self.tree.heading("Select", text="Select")
        self.tree.heading("Path", text="Path")
        self.tree.heading("Size", text="Size")
        self.tree.heading("Days", text="Days")
        self.tree.heading("Action", text="Action")
        
        self.tree.pack(expand=True, fill="both", padx=5, pady=5)
        scrollbar.configure(command=self.tree.yview)
        self.tree.bind("<Button-1>", self.toggle_selection)

    def check_queue(self):
        """Checks the queue periodically to update the GUI from the background thread"""
        try:
            while True:
                message = self.result_queue.get_nowait()
                self.handle_queue_message(message)
                self.result_queue.task_done()
        except queue.Empty:
            pass
        
        # Schedule next check after 100ms
        self.root.after(100, self.check_queue)
    
    def handle_queue_message(self, message):
        """Processes messages from the background thread"""
        msg_type = message.get('type')
        
        if msg_type == 'status':
            self.status_label.configure(text=message.get('text', ''))
        elif msg_type == 'progress':
            progress = message.get('value', 0)
            if progress > 0 and self.is_processing:
                self.progress_bar.pack(side="right", padx=10, pady=8)
                self.progress_bar['value'] = progress
            elif not self.is_processing:
                self.progress_bar.pack_forget()
        elif msg_type == 'analyze_complete':
            self.update_analyze_results(message)
        elif msg_type == 'execute_complete':
            self.update_execute_results(message)
        elif msg_type == 'error':
            messagebox.showerror("Error", message.get('text', 'An error occurred'))
            self.set_processing(False)
    
    def set_processing(self, is_processing):
        """Enable/disable processing status"""
        self.is_processing = is_processing
        state = "disabled" if is_processing else "normal"
        
        self.analyze_btn.configure(state=state)
        self.browse_btn.configure(state=state)
        self.history_btn.configure(state=state)

        # Only enable Execute button if not processing AND there are results
        execute_state = "normal" if not is_processing and self.tree.get_children() else "disabled"
        self.execute_btn.configure(state=execute_state)
        
        if not is_processing:
            self.status_label.configure(text="Ready")
            self.progress_bar.pack_forget()
            self.progress_bar['value'] = 0
            
    def start_analyze(self):
        """Starts analysis in a background thread"""
        if self.is_processing:
            return
            
        folder_str = self.folder_path.get()
        folder = Path(folder_str)

        if not folder.is_dir():
            messagebox.showerror("Error", "Invalid folder.")
            return

        if self.is_critical_path(folder_str):
            messagebox.showerror("Permission Error", "Permission denied.")
            return
        
        self.tree.delete(*self.tree.get_children()) # Clear old results
        self.summary_label.configure(text="")
        self.set_processing(True)
        self.status_label.configure(text="Scanning folder...")
        self.progress_bar.pack(side="right", padx=10, pady=8)
        self.progress_bar['value'] = 0.1
        
        # Run in a separate thread
        thread = threading.Thread(target=self.analyze_thread, args=(folder,), daemon=True)
        thread.start()
    
    def start_execute_actions(self):
        """Starts action execution in a background thread"""
        if self.is_processing:
            return
        
        # Get the list of items to process from the treeview (in main thread)
        items_to_process = []
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            if values and values[0] == "✓": # Selected
                items_to_process.append({
                    'item': item,
                    'path': values[1],
                    'action': values[4]
                })
        
        if not items_to_process:
            messagebox.showinfo("Notification", "Please select at least one file to process.")
            return
        
        confirm = messagebox.askyesno(
            "Confirm Action",
            f"Are you sure you want to perform cleanup actions for {len(items_to_process)} files?"
        )
        
        if not confirm:
            return
        
        self.set_processing(True)
        self.status_label.configure(text="Executing actions...")
        self.progress_bar.pack(side="right", padx=10, pady=8)
        self.progress_bar['value'] = 0.1
        
        # Run in a separate thread with the data already read
        thread = threading.Thread(target=self.execute_actions_thread, args=(items_to_process,), daemon=True)
        thread.start()

    def is_critical_path(self, folder_str):
        """Checks if the path is in CRITICAL_PATHS (optimized)."""
        folder_str = str(Path(folder_str).resolve())
        
        # Check for exact match first
        if folder_str in CRITICAL_PATHS_SET:
            return True
        
        # Check for prefix
        for cp in CRITICAL_PATHS_SET:
            if folder_str.startswith(cp + os.sep) and cp != HOME_DIR: # Add condition to avoid checking home dir as critical path
                return True
        return False

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            # Protection layer 1: Block critical folders immediately upon selection
            if self.is_critical_path(folder):
                messagebox.showerror(
                    "Permission Error", 
                    "Permission denied."
                )
            else:
                self.folder_path.set(folder)

    def toggle_selection(self, event):
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if item and col == "#1":
            values = list(self.tree.item(item, "values"))
            # Toggle between "✓" and ""
            values[0] = "✓" if values[0] == "" else "" 
            self.tree.item(item, values=values)
    
    # ... (analyze_thread, scan_folder, calculate_features, format_size remain largely the same as they are internal logic or data processing) ...

    def analyze_thread(self, folder):
        """Runs analysis in a background thread"""
        try:
            self.result_queue.put({'type': 'status', 'text': 'Scanning folder...'})
            self.result_queue.put({'type': 'progress', 'value': 0.2})
            
            df, error_msg = self.scan_folder(folder)
            
            if error_msg:
                self.result_queue.put({'type': 'error', 'text': 'Permission denied.'})
                return
                
            if df.empty:
                self.result_queue.put({
                    'type': 'analyze_complete',
                    'success': False,
                    'message': 'No eligible files found.'
                })
                return

            self.result_queue.put({'type': 'status', 'text': f'Calculating features for {len(df)} files...'})
            self.result_queue.put({'type': 'progress', 'value': 0.4})
            
            df = self.calculate_features(df)
            
            feature_cols = ['size_log', 'days_since_access', 'is_temp_file',
                            'num_words', 'name_length', 'has_temp_keyword', 'has_important_keyword']
            embedding_cols = [col for col in df.columns if col.startswith('embedding_dim_')]
            feature_cols.extend(embedding_cols)
            
            available_cols = [col for col in feature_cols if col in df.columns]
            X = df[available_cols]
            
            self.result_queue.put({'type': 'status', 'text': 'Predicting with ML...'})
            self.result_queue.put({'type': 'progress', 'value': 0.6})
            
            # Process prediction in larger batches for speed
            preds_list = []
            pred_batch_size = 5000 
            total_batches = (len(X) + pred_batch_size - 1) // pred_batch_size
            
            for idx, pred_start in enumerate(range(0, len(X), pred_batch_size)):
                pred_end = min(pred_start + pred_batch_size, len(X))
                batch_X = X.iloc[pred_start:pred_end]
                # Ensure input is numpy array if needed (joblib/sklearn usually handles DataFrame automatically)
                batch_preds = self.model.predict(batch_X) 
                preds_list.extend(batch_preds)
                
                # Update progress
                progress = 0.6 + 0.2 * (idx + 1) / total_batches
                self.result_queue.put({'type': 'progress', 'value': progress})
                
                del batch_X, batch_preds
                if idx % 5 == 0: 
                    gc.collect()
                    
            preds = np.array(preds_list)
            del preds_list
            
            df['Predicted_Label'] = self.encoder.inverse_transform(preds)
            df['Formatted_Size'] = df['size_bytes'].apply(self.format_size)

            self.result_queue.put({'type': 'status', 'text': 'Analysis complete!'})
            self.result_queue.put({'type': 'progress', 'value': 0.9})
            
            # Send results back to the main thread
            result_df = df[df['Predicted_Label'] != 'Keep'].copy()
            total_size_clean = result_df['size_bytes'].sum()
            
            tree_data = []
            for _, row in result_df.iterrows():
                tree_data.append({
                    # Default selection is empty (user will select)
                    'select': "", 
                    'file_path': row['file_path'],
                    'size': row['Formatted_Size'],
                    'days': f"{int(row['days_since_access'])} days",
                    'action': row['Predicted_Label']
                })
            
            self.result_queue.put({
                'type': 'analyze_complete',
                'success': True,
                'tree_data': tree_data,
                'total_size_clean': total_size_clean,
                'has_results': len(tree_data) > 0
            })
            
            # Free memory
            del X, preds, df, result_df
            gc.collect()
            
        except Exception as e:
            error_msg = str(e)
            self.result_queue.put({'type': 'error', 'text': f'An error occurred during analysis:\n{error_msg}'})
    
    def update_analyze_results(self, message):
        """Updates analysis results in the GUI"""
        
        self.set_processing(False) # Reset processing status before updating GUI
        
        if not message.get('success'):
            messagebox.showinfo("Notification", message.get('message', ''))
            self.tree.delete(*self.tree.get_children())
            self.summary_label.configure(text="")
            return
        
        tree_data = message.get('tree_data', [])
        total_size_clean = message.get('total_size_clean', 0)
        
        # Clear old data
        self.tree.delete(*self.tree.get_children())
        
        # Add new data
        for item in tree_data:
            self.tree.insert("", "end", values=(
                item['select'], 
                item['file_path'], 
                item['size'], 
                item['days'], 
                item['action']
            ))
        
        if message.get('has_results'):
            formatted_size = self.format_size(total_size_clean)
            self.summary_label.configure(
                text=f"✅ Suggested cleanup for {len(tree_data)} files, total size: **{formatted_size}**"
            )
            # Enable Execute button
            self.execute_btn.configure(state="normal") 
        else:
            messagebox.showinfo("Notification", "The model did not suggest any actions for the scanned files.")
            self.summary_label.configure(text="No files suggested for cleanup.")
            self.execute_btn.configure(state="disabled") # Disable Execute button


    def scan_folder(self, folder):
        """Optimized folder scanning"""
        now = time.time()
        data = []
        folder_str = str(folder)
        
        # CRITICAL_PATHS protection layer 
        if self.is_critical_path(folder_str):
            return pd.DataFrame(), "Permission denied." 
            
        try:
            # Check access permission
            list(folder.iterdir()) 
        except PermissionError:
            return pd.DataFrame(), "Permission denied."
        except Exception:
            return pd.DataFrame(), None 

        DAYS_THRESHOLD = 7 * 86400 # Convert to seconds
        
        for path in folder.rglob("*"):
            # Skip critical paths early
            if self.is_critical_path(str(path)):
                continue
                
            if path.is_file():
                try:
                    stat = path.stat()
                    days_seconds = now - stat.st_mtime
                    
                    # Skip files newer than 7 days
                    if days_seconds < DAYS_THRESHOLD:
                        continue
                        
                    ext = path.suffix.lower()
                    
                    # Use math.log10 to avoid numpy error
                    size_log_val = math.log10(stat.st_size + 1)
                    
                    data.append({
                        "file_path": str(path),
                        "size_bytes": stat.st_size,
                        "extension": ext,
                        "days_since_access": days_seconds / 86400,
                        "is_temp_file": int(ext in TEMP_EXTS_SET),
                        "size_log": size_log_val
                    })
                except (PermissionError, OSError):
                    continue
                except Exception:
                    continue
                    
        return pd.DataFrame(data), None

    def calculate_features(self, df):
        """Simplified feature calculation - optimized for speed"""
        if df.empty: 
            return df

        # Vectorized operations (maintained)
        df['is_temp_file'] = df['extension'].isin(TEMP_EXTS).astype(int)
        df['size_log'] = np.log10(df['size_bytes'] + 1)
        
        # Extract simple features in a larger batch
        BATCH_SIZE = 1000 
        total_files = len(df)
        
        features_list = []
        embedding_list = []
        
        for batch_start in range(0, total_files, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_files)
            
            # Process batch
            for file_path in df.iloc[batch_start:batch_end]['file_path']:
                try:
                    features = extract_simple_features(file_path)
                    features_list.append({
                        'num_words': features['num_words'],
                        'name_length': features['name_length'],
                        'has_temp_keyword': features['has_temp_keyword'],
                        'has_important_keyword': features['has_important_keyword']
                    })
                    embedding_list.append(features['embedding_vector'])
                except:
                    # Fallback values
                    features_list.append({
                        'num_words': 0,
                        'name_length': len(Path(file_path).stem),
                        'has_temp_keyword': 0,
                        'has_important_keyword': 0
                    })
                    embedding_list.append(np.zeros(684, dtype=np.float32))
            
            # Cleanup after a large batch
            if batch_end % 5000 == 0:
                gc.collect()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list, index=df.index)
        df = pd.concat([df, features_df], axis=1)
        
        # Process embedding with PCA
        embedding_matrix = np.array(embedding_list, dtype=np.float32)
        
        # PCA transform - process all at once if possible
        if len(embedding_matrix) <= 5000:
            embedding_reduced = self.pca.transform(embedding_matrix)
        else:
            # Split for PCA if too large
            pca_results = []
            for pca_start in range(0, len(embedding_matrix), 3000):
                pca_end = min(pca_start + 3000, len(embedding_matrix))
                pca_results.append(self.pca.transform(embedding_matrix[pca_start:pca_end]))
            embedding_reduced = np.vstack(pca_results)
        
        # Add embedding columns
        embedding_cols = [f'embedding_dim_{i}' for i in range(embedding_reduced.shape[1])]
        embedding_df = pd.DataFrame(embedding_reduced, columns=embedding_cols, index=df.index)
        df = pd.concat([df, embedding_df], axis=1)
        
        # Cleanup
        del embedding_matrix, embedding_list, features_list, embedding_reduced, embedding_df, features_df
        gc.collect()
        
        return df

    def format_size(self, size):
        if size >= 1024**3:
            return f"{size / 1024**3:.2f} GB"
        elif size >= 1024**2:
            return f"{size / 1024**2:.2f} MB"
        elif size >= 1024:
            return f"{size / 1024:.2f} KB"
        else:
            return f"{size} bytes"

    def execute_actions_thread(self, items_to_process):
        """Runs action execution in a background thread"""
        try:
            # Fix artifact directory to be in the home directory
            base_artifact_dir = Path.home() / CLEANER_ARTIFACTS_DIR
            
            try:
                base_artifact_dir.mkdir(exist_ok=True)
            except Exception as e:
                self.result_queue.put({
                    'type': 'error',
                    'text': f'Could not create archive directory at {base_artifact_dir}. Details: {e}'
                })
                return
                
            archive_dir = base_artifact_dir / "ARCHIVE_ML_ASSISTANT"
            archive_dir.mkdir(exist_ok=True)
            
            total_items = len(items_to_process)
            count = 0
            has_permission_error = False
            items_to_remove = []
            
            self.result_queue.put({'type': 'status', 'text': f'Processing {total_items} files...'})
            
            # Process each item
            for idx, item_data in enumerate(items_to_process):
                file_path = Path(item_data['path'])
                action = item_data['action']
                
                # Update progress
                progress = 0.1 + 0.8 * (idx + 1) / total_items
                self.result_queue.put({'type': 'progress', 'value': progress})
                self.result_queue.put({
                    'type': 'status',
                    'text': f'Processing {idx + 1}/{total_items}: {file_path.name}'
                })
                
                # FINAL PROTECTION LAYER (CRITICAL PATHS)
                if self.is_critical_path(str(file_path)):
                    continue 

                try:
                    success = False
                    status = "Failed"
                    if action == "Delete":
                        send2trash(file_path)
                        status = "Deleted"
                        success = True
                        
                    elif action == "Compress":
                        zip_path = archive_dir / f"{file_path.stem}.zip"
                        # Ensure unique archive name
                        if zip_path.exists():
                            zip_path = archive_dir / f"{file_path.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
                            
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            # Write file to the archive, with original filename inside
                            zipf.write(file_path, file_path.name) 
                        file_path.unlink()
                        status = "Compressed and deleted original"
                        success = True
                        
                    if success:
                        self.history.append({
                            "file": str(file_path),
                            "action": action,
                            "status": status,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        count += 1
                        items_to_remove.append(item_data['item'])
                        
                except PermissionError:
                    has_permission_error = True
                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")
            
            # Send results back to main thread
            self.result_queue.put({
                'type': 'execute_complete',
                'success': True,
                'count': count,
                'has_permission_error': has_permission_error,
                'items_to_remove': items_to_remove
            })
            
        except Exception as e:
            self.result_queue.put({
                'type': 'error',
                'text': f'An error occurred during action execution:\n{str(e)}'
            })
    
    def update_execute_results(self, message):
        """Updates action execution results in the GUI"""
        count = message.get('count', 0)
        has_permission_error = message.get('has_permission_error', False)
        items_to_remove = message.get('items_to_remove', [])
        
        # Remove successfully processed items from the treeview
        for item in items_to_remove:
            self.tree.delete(item)
        
        # Update summary
        if self.tree.get_children():
            # Update remaining file count and size (if calculable quickly)
            self.summary_label.configure(text=f"Processing complete. {len(self.tree.get_children())} files remaining.")
        else:
            self.summary_label.configure(text="✅ Cleanup Complete!")

        # Display results
        if has_permission_error:
            messagebox.showerror("Permission Error", "Permission denied.")
        
        if count > 0:
            messagebox.showinfo("Complete", f"Successfully executed {count} actions.")
        elif not has_permission_error:
            messagebox.showinfo("Complete", "0 actions executed. Please check file selection or permission errors.")
        
        self.set_processing(False)
            
    def show_history(self):
        if not self.history:
            messagebox.showinfo("History", "No actions recorded yet.")
            return
        
        hist_win = tk.Toplevel(self.root)
        hist_win.title("Action History")
        hist_win.geometry("900x500")
        
        # Title
        title_label = ttk.Label(hist_win, text="Action History", font=("Arial", 14, "bold"))
        title_label.pack(pady=(15, 10))
        
        # Frame containing treeview
        tree_frame = ttk.Frame(hist_win)
        tree_frame.pack(expand=True, fill="both", padx=20, pady=(0, 20))
        
        # Scrollbar and Treeview
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")
        
        tree = ttk.Treeview(
            tree_frame, 
            columns=("File", "Action", "Status", "Time"), 
            show="headings", 
            yscrollcommand=scrollbar.set
        )
        tree.column("File", width=400)
        tree.column("Action", width=150, anchor="center")
        tree.column("Status", width=250)
        tree.column("Time", width=150)
        
        for col in tree["columns"]:
            tree.heading(col, text=col)
        
        tree.pack(expand=True, fill="both", padx=5, pady=5)
        scrollbar.configure(command=tree.yview)
        
        for row in self.history:
            tree.insert("", "end", values=(row["file"], row["action"], row["status"], row["time"]))

# APPLICATION START
if __name__ == "__main__":
    # Ensure assets directory exists (for model files)
    Path(ASSETS_DIR).mkdir(exist_ok=True) 
    
    root = tk.Tk()
    app = DiskCleanerApp(root)
    root.mainloop()
