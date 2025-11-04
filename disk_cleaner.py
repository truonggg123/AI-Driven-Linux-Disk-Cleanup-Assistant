import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import pandas as pd
import joblib
import time
import numpy as np
import zipfile
import os
import sys
from datetime import datetime
from send2trash import send2trash
import spacy
from transformers import AutoTokenizer, AutoModel
import torch

# CÁC BIẾN HẰNG
ASSETS_DIR = Path("ml_assets")
MODEL_FILE = ASSETS_DIR / "disk_model.joblib"
ENCODER_FILE = ASSETS_DIR / "label_encoder.joblib"
PCA_FILE = ASSETS_DIR / "pca_transformer.joblib"  # PCA transformer cho embedding vectors
TEMP_EXTS = [".log", ".tmp", ".bak", ".cache", ".~", ""]
RECYCLE_DIR_NAME = "RECYCLE_BIN_ML_ASSISTANT"
TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lấy thư mục người dùng hiện tại (cần thiết để chặn các thư mục ẩn)
HOME_DIR = str(Path.home())

# DANH SÁCH ĐEN CÁC THƯ MỤC HỆ THỐNG QUAN TRỌNG TRÊN LINUX
CRITICAL_PATHS = [

    "/",

 # Cốt lõi hệ thống & nhị phân
    "/bin", "/sbin", "/lib", "/lib64", 
    "/usr/bin", "/usr/sbin", "/usr/lib", "/usr/lib64", 
    "/usr/local", "/opt",
    
    # Cấu hình & Dữ liệu hệ thống
    "/etc", "/boot", "/root", 
    
    # Thư mục ảo & Thiết bị
    "/dev", "/proc", "/sys", "/run", 
    
    # Thư mục gắn kết & Cache hệ thống
    "/mnt", "/media", "/snap", "/lost+found",
    
    # Dữ liệu trạng thái thời gian chạy & LOGS HỆ THỐNG
    "/var/run", "/var/lock", 
    "/var/log", 
    
    # THƯ MỤC CẤU HÌNH CÁ NHÂN QUAN TRỌNG (Sử dụng đường dẫn tuyệt đối)
    f"{HOME_DIR}/.ssh",
    f"{HOME_DIR}/.config",
    f"{HOME_DIR}/.local",
]

# ----------------- HÀM TRÍCH XUẤT EMBEDDING TỪ TRANSFORMERS -----------------
def get_transformer_embedding(text, tokenizer, model, max_length=128):
    """
    Lấy embedding từ transformer model.
    """
    if tokenizer is None or model is None:
        return None
    
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, 
                          truncation=True, padding=True)
        
        # Lấy embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Lấy mean pooling của token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        return embeddings.numpy()
    except Exception as e:
        return None

# ----------------- HÀM TRÍCH XUẤT FEATURES TỪ TÊN FILE BẰNG SPACY + TRANSFORMERS -----------------
def extract_spacy_features(file_path, nlp_model, transformer_tokenizer=None, transformer_model=None):
    """
    Trích xuất features từ tên file sử dụng kết hợp spaCy và Transformers.
    Trả về: dict chứa các features từ NLP
    """
    # Lấy tên file không có extension
    file_name = Path(file_path).stem.lower()
    
    # Xử lý bằng spaCy
    doc = nlp_model(file_name)
    
    # Feature 1: Số lượng từ trong tên file
    num_words = len([token for token in doc if token.is_alpha])
    
    # Feature 2: Độ dài tên file
    name_length = len(file_name)
    
    # Feature 3: Có chứa từ khóa temp/old/backup không
    temp_keywords = ['temp', 'tmp', 'cache', 'old', 'backup', 'bak', '~']
    has_temp_keyword = any(keyword in file_name for keyword in temp_keywords)
    
    # Feature 4: Có chứa từ khóa quan trọng không
    important_keywords = ['important', 'final', 'document', 'report']
    has_important_keyword = any(keyword in file_name for keyword in important_keywords)
    
    # Feature 5: Embedding vector từ spaCy (300D từ en_core_web_lg)
    if len(doc) > 0 and doc.vector is not None:
        spacy_embedding = doc.vector  # Vector 300D từ en_core_web_lg
    else:
        spacy_embedding = np.zeros(300)  # Vector 0 nếu không có token
    
    # Feature 6: Embedding vector từ Transformers (384D từ all-MiniLM-L6-v2)
    transformer_embedding = get_transformer_embedding(file_name, transformer_tokenizer, transformer_model)
    if transformer_embedding is None or len(transformer_embedding) != 384:
        transformer_embedding = np.zeros(384)  # Vector 0 nếu không có transformer
    
    # Kết hợp cả hai embeddings (sẽ giảm chiều bằng PCA sau)
    # Luôn luôn 684D: 300D (spaCy) + 384D (Transformers)
    combined_embedding = np.concatenate([spacy_embedding, transformer_embedding])  # 300 + 384 = 684D
    
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
        self.root.title("Trợ lý Dọn Dẹp Đĩa ML")
        self.root.geometry("900x600")

        # Khối try-except cho việc tải mô hình
        try:
            self.model = joblib.load(MODEL_FILE)
            self.encoder = joblib.load(ENCODER_FILE)
            self.pca = joblib.load(PCA_FILE)
            print(f"[Khởi tạo] Đã tải mô hình từ: {MODEL_FILE}")
            print(f"[Khởi tạo] Đã tải PCA transformer từ: {PCA_FILE}")
        except FileNotFoundError as e:
            messagebox.showerror("Lỗi Khởi tạo", 
                               f"Không tìm thấy tệp mô hình ML hoặc PCA transformer.\n"
                               f"Hãy đảm bảo thư mục 'ml_assets' tồn tại.\n"
                               f"Chi tiết: {e}")
            self.root.destroy()
            return
        
        # Khởi tạo spaCy model
        try:
            self.nlp = spacy.load("en_core_web_lg")
            print("[Khởi tạo] Đã tải mô hình spaCy: en_core_web_lg")
        except OSError:
            messagebox.showerror("Lỗi Khởi tạo", 
                               "Không tìm thấy mô hình spaCy 'en_core_web_lg'.\n"
                               "Vui lòng chạy: python -m spacy download en_core_web_lg")
            self.root.destroy()
            return
        
        # Khởi tạo Transformers model (sử dụng mô hình nhẹ cho text embeddings)
        try:
            print(f"[Khởi tạo] Đang tải mô hình Transformers: {TRANSFORMER_MODEL_NAME}...")
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
            self.transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
            self.transformer_model.eval()  # Chế độ evaluation
            print("[Khởi tạo] Đã tải mô hình Transformers thành công")
        except Exception as e:
            print(f"[Cảnh báo] Không thể tải mô hình Transformers: {e}")
            print("Chương trình sẽ tiếp tục chỉ sử dụng spaCy")
            self.transformer_tokenizer = None
            self.transformer_model = None

        self.folder_path = tk.StringVar()
        self.safe_delete = tk.BooleanVar(value=True)
        self.history = []

        self.create_widgets()

    def create_widgets(self):
        # --- Chọn thư mục ---
        path_frame = tk.Frame(self.root)
        path_frame.pack(pady=10)

        tk.Label(path_frame, text="Chọn thư mục:").grid(row=0, column=0, padx=5)
        tk.Entry(path_frame, textvariable=self.folder_path, width=60).grid(row=0, column=1)
        tk.Button(path_frame, text="Duyệt...", command=self.browse_folder).grid(row=0, column=2, padx=5)

        # --- Tùy chọn xóa ---
        options_frame = tk.Frame(self.root)
        options_frame.pack()
        tk.Checkbutton(options_frame, text="Xóa an toàn (gửi vào thùng rác)", variable=self.safe_delete).pack()

        # --- Nhóm nút chức năng ---
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Phân tích", width=20, command=self.analyze).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Thực hiện hành động", width=20, command=self.execute_actions).grid(row=0, column=1, padx=5)
        tk.Button(button_frame, text="Xem lịch sử", width=20, command=self.show_history).grid(row=0, column=2, padx=5)
        tk.Button(button_frame, text="Khôi phục file đã xóa", width=20, command=self.restore_files).grid(row=0, column=3, padx=5)

        # --- Treeview hiển thị danh sách ---
        tree_frame = tk.Frame(self.root)
        tree_frame.pack(expand=True, fill="both", padx=10, pady=10)

        scrollbar = tk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")

        self.tree = ttk.Treeview(tree_frame, columns=("Chọn", "Path", "Size", "Days", "Action"), show="headings", yscrollcommand=scrollbar.set)
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
        self.tree.pack(expand=True, fill="both")
        scrollbar.config(command=self.tree.yview)
        self.tree.bind("<Button-1>", self.toggle_selection)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)

    def toggle_selection(self, event):
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if item and col == "#1":
            values = list(self.tree.item(item, "values"))
            values[0] = "❌" if values[0] == "" else ""
            self.tree.item(item, values=values)

    def analyze(self):
        folder = Path(self.folder_path.get())
        if not folder.is_dir():
            messagebox.showerror("Lỗi", "Thư mục không hợp lệ.")
            return

        self.df = self.scan_folder(folder)
        if self.df.empty:
            messagebox.showinfo("Thông báo", "Không có tệp nào đủ điều kiện.")
            return

        # Tính toán features NLP đầy đủ
        self.df = self.calculate_features(self.df)
        
        # Chọn các cột feature mà mô hình đã được huấn luyện
        feature_cols = ['size_log', 'days_since_access', 'is_temp_file',
                        'num_words', 'name_length', 'has_temp_keyword', 'has_important_keyword']
        # Thêm các embedding dimensions (số lượng có thể thay đổi tùy theo PCA)
        embedding_cols = [col for col in self.df.columns if col.startswith('embedding_dim_')]
        feature_cols.extend(embedding_cols)
        
        # Chỉ lấy các cột có trong DataFrame (tránh lỗi nếu thiếu)
        available_cols = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_cols]
        
        # Dự đoán bằng mô hình ML
        preds = self.model.predict(X)
        self.df['Predicted_Label'] = self.encoder.inverse_transform(preds)
        self.df['Formatted_Size'] = self.df['size_bytes'].apply(self.format_size)

        self.tree.delete(*self.tree.get_children())
        # Chỉ hiển thị các tệp mà mô hình dự đoán là hành động ('Delete' hoặc 'Compress')
        for _, row in self.df[self.df['Predicted_Label'] != 'Keep'].iterrows():
            self.tree.insert("", "end", values=("", row['file_path'], row['Formatted_Size'], f"{int(row['days_since_access'])} ngày", row['Predicted_Label']))

    def scan_folder(self, folder):
        now = time.time()
        data = []
        
        folder_str = str(folder)
        
        # CHÈN LỚP BẢO VỆ 1: CHẶN THƯ MỤC GỐC ĐƯỢC CHỌN (ĐÃ FIX)
        is_critical_path = False
        for cp in CRITICAL_PATHS:
            # Kiểm tra khớp hoàn toàn HOẶC bắt đầu bằng CP + "/" (là thư mục con)
            if folder_str == cp or folder_str.startswith(cp + "/"):
                is_critical_path = True
                break
        
        if is_critical_path:
            messagebox.showwarning("Cảnh báo Hệ thống", f"Thư mục đã chọn '{folder_str}' nằm trong một thư mục hệ thống quan trọng và sẽ không được quét.")
            return pd.DataFrame()
            
        for path in folder.rglob("*"):
            path_str = str(path)
            
            # CHÈN LỚP BẢO VỆ 2: LỌC TỆP CON (ĐÃ FIX)
            is_critical_file = False
            for cp in CRITICAL_PATHS:
                # Kiểm tra khớp hoàn toàn HOẶC bắt đầu bằng CP + "/"
                if path_str == cp or path_str.startswith(cp + "/"):
                    is_critical_file = True
                    break
            
            if is_critical_file:
                continue
                
            if path.is_file():
                try:
                    stat = path.stat()
                    
                    # Dùng st_mtime (thời gian sửa đổi) cho tính chính xác cao hơn
                    days = (now - stat.st_mtime) / (24 * 3600)
                    
                    # Bỏ qua tệp được sửa đổi trong vòng 7 ngày (lớp bảo vệ hoạt động)
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
                    # Bỏ qua các tệp mà người dùng không có quyền đọc (thường là tệp hệ thống)
                    continue
                except Exception:
                    # Bỏ qua các lỗi I/O khác
                    continue
        return pd.DataFrame(data)

    def calculate_features(self, df):
        """
        Tính toán Features (đặc trưng) cần thiết cho mô hình.
        Bao gồm các features từ spaCy + Transformers giống như trong train_model.py
        """
        if df.empty:
            return df

        # Đặc trưng 1: Là file tạm thời/rác hay không
        df['is_temp_file'] = df['extension'].isin(TEMP_EXTS).astype(int)

        # Đặc trưng 2: Kích thước tệp (dùng log scale như khi train)
        df['size_log'] = np.log10(df['size_bytes'] + 1)
        
        # Trích xuất features từ tên file bằng spaCy + Transformers
        print("   Đang trích xuất features từ tên file bằng spaCy + Transformers...")
        spacy_features = []
        embedding_vectors = []
        
        for idx, file_path in enumerate(df['file_path']):
            features = extract_spacy_features(file_path, self.nlp, self.transformer_tokenizer, self.transformer_model)
            spacy_features.append({
                'num_words': features['num_words'],
                'name_length': features['name_length'],
                'has_temp_keyword': features['has_temp_keyword'],
                'has_important_keyword': features['has_important_keyword']
            })
            embedding_vectors.append(features['embedding_vector'])
            
            if (idx + 1) % 1000 == 0:
                print(f"   Đã xử lý {idx + 1}/{len(df)} tệp...")
        
        # Thêm các features từ spaCy vào DataFrame
        spacy_df = pd.DataFrame(spacy_features)
        df = pd.concat([df, spacy_df], axis=1)
        
        # Giảm chiều embedding vectors bằng PCA đã được huấn luyện
        embedding_matrix = np.array(embedding_vectors)
        embedding_reduced = self.pca.transform(embedding_matrix)
        
        # Tạo tên cột cho embedding features
        embedding_cols = [f'embedding_dim_{i}' for i in range(embedding_reduced.shape[1])]
        embedding_df = pd.DataFrame(embedding_reduced, columns=embedding_cols, index=df.index)
        
        # Kết hợp tất cả features
        df = pd.concat([df, embedding_df], axis=1)
        
        return df

    def format_size(self, size):
        if size >= 1024**3:
            return f"{size / 1024**3:.2f} GB"
        elif size >= 1024**2:
            return f"{size / 1024**2:.2f} MB"
        else:
            return f"{size / 1024:.2f} KB"

    def execute_actions(self):
        folder = Path(self.folder_path.get())
        archive_dir = folder / "ARCHIVE_ML_ASSISTANT"
        archive_dir.mkdir(exist_ok=True)
        recycle_dir = folder / RECYCLE_DIR_NAME
        recycle_dir.mkdir(exist_ok=True)

        count = 0
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            # Chỉ xử lý các mục được người dùng chọn bằng "❌"
            if values[0] != "❌":
                continue
            
            path, _, _, action = values[1], values[2], values[3], values[4]
            file_path = Path(path)
            
            # CHÈN LỚP BẢO VỆ 3: LỚP BẢO VỆ CUỐI CÙNG (ĐÃ FIX)
            path_str = str(file_path)
            is_critical_action = False
            for cp in CRITICAL_PATHS:
                # Kiểm tra khớp hoàn toàn HOẶC bắt đầu bằng CP + "/"
                if path_str == cp or path_str.startswith(cp + "/"):
                    is_critical_action = True
                    break
            
            if is_critical_action:
                print(f"Bảo vệ: Bỏ qua hành động trên tệp hệ thống quan trọng: {file_path}")
                continue 

            try:
                if action == "Delete":
                    if self.safe_delete.get():
                        send2trash(file_path)
                        status = "Xóa an toàn"
                    else:
                        file_path.rename(recycle_dir / file_path.name)
                        status = "Di chuyển vào thùng rác nội bộ"
                elif action == "Compress":
                    zip_path = archive_dir / f"{file_path.stem}.zip"
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(file_path, file_path.name) 
                    file_path.unlink()
                    status = "Nén và xóa gốc"
                
                self.history.append({
                    "file": str(file_path),
                    "action": action,
                    "status": status,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                count += 1
            except Exception as e:
                print(f"Lỗi xử lý {file_path.name}: {e}")

        messagebox.showinfo("Hoàn tất", f"Đã thực hiện {count} hành động.")
        self.analyze()

    def show_history(self):
        if not self.history:
            messagebox.showinfo("Lịch sử", "Chưa có hành động nào.")
            return
        hist_win = tk.Toplevel(self.root)
        hist_win.title("Lịch sử hành động")
        tree = ttk.Treeview(hist_win, columns=("File", "Action", "Status", "Time"), show="headings")
        for col in tree["columns"]:
            tree.heading(col, text=col)
        tree.pack(expand=True, fill="both")
        for row in self.history:
            tree.insert("", "end", values=(row["file"], row["action"], row["status"], row["time"]))

    def restore_files(self):
        folder = Path(self.folder_path.get())
        recycle_dir = folder / RECYCLE_DIR_NAME
        if not recycle_dir.exists():
            messagebox.showinfo("Khôi phục", "Không có file nào để khôi phục.")
            return
        files = list(recycle_dir.glob("*"))
        if not files:
            messagebox.showinfo("Khôi phục", "Thùng rác nội bộ trống.")
            return
        
        messagebox.showinfo("Khôi phục", f"Tìm thấy {len(files)} file trong Thùng rác nội bộ.\nCần phát triển giao diện Khôi phục chi tiết.")

# KHỞI CHẠY ỨNG DỤNG
if __name__ == "__main__":
    root = tk.Tk()
    app = DiskCleanerApp(root)
    root.mainloop()
