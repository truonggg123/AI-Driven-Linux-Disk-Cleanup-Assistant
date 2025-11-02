import os
import pandas as pd
import numpy as np
import random 
import joblib # Thư viện để lưu và tải mô hình
from pathlib import Path

# Thư viện cho Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ==================== CẤU HÌNH DỮ LIỆU GIẢ VÀ LUẬT GÁN NHÃN ====================
# Thư mục lưu trữ mô hình đã huấn luyện
ASSETS_DIR = Path("ml_assets")
MODEL_FILE = ASSETS_DIR / "disk_model.joblib"
ENCODER_FILE = ASSETS_DIR / "label_encoder.joblib"

# Cấu hình cho việc tạo dữ liệu giả (CHỈ DÙNG CHO TRAINING)
SYNTHETIC_NUM_FILES = 5000 # Tăng số lượng mẫu giả để huấn luyện tốt hơn
SYNTHETIC_MAX_SIZE_MB = 200
SYNTHETIC_MAX_AGE_DAYS = 730
SYNTHETIC_EXTENSIONS = [ 
    ".pdf", ".docx", ".xlsx", ".zip", ".rar", ".7z", 
    ".jpg", ".png", ".mp4", ".mov", ".gif", ".iso", 
    ".log", ".tmp", ".bak", ".cache", ".~",  
    ".py", ".sh", ".c", ".cpp", ".h", ".html", 
    ".txt", ".js", ".json", ".xml", ".db", ""
]

# Các ngưỡng này định nghĩa Ground Truth cho mô hình
DELETE_SIZE_THRESHOLD = 5 * 1024 * 1024   # 5MB
DELETE_TIME_THRESHOLD = 180               # 180 ngày

COMPRESS_SIZE_THRESHOLD = 50 * 1024 * 1024 # 50MB
COMPRESS_TIME_THRESHOLD = 90              # 90 ngày
COMPRESSED_EXTS = [".zip", ".rar", ".7z", ".tar.gz", ".gz"]
# ==============================================================================


# ----------------- HÀM TẠO METADATA GIẢ TRONG BỘ NHỚ -----------------
def generate_synthetic_metadata():
    """Tạo metadata giả trong bộ nhớ để huấn luyện mô hình."""
    file_data_list = []
    
    print(f"\n[BƯỚC 1] Bắt đầu tạo {SYNTHETIC_NUM_FILES} metadata giả trong bộ nhớ...")

    for i in range(SYNTHETIC_NUM_FILES):
        ext = random.choice(SYNTHETIC_EXTENSIONS)
        # Kích thước ngẫu nhiên theo log scale
        size_bytes = int(10 ** (random.uniform(2, np.log10(SYNTHETIC_MAX_SIZE_MB * 1024 * 1024)))) 
        days_ago = random.randint(1, SYNTHETIC_MAX_AGE_DAYS) 
        
        file_data_list.append({
            'file_path': f"synthetic_file_{i}{ext}",
            'size_bytes': size_bytes,
            'extension': ext,
            'days_since_access': days_ago,
        })
    
    synthetic_df = pd.DataFrame(file_data_list)
    return label_data(synthetic_df) 

# -------------------- HÀM XỬ LÝ VÀ GÁN NHÃN (Sử dụng để tính Feature) --------------------
def label_data(df):
    """Tính toán Features (đặc trưng) và gán nhãn Ground Truth."""
    if df.empty:
        return df

    temp_extensions = [".log", ".tmp", ".bak", ".cache", ".~", ""]
    df['is_temp_file'] = df['extension'].isin(temp_extensions).astype(int)

    # Sử dụng log10 cho kích thước
    df['size_log'] = np.log10(df['size_bytes'] + 1)
    
    df['Label'] = 'Keep'

    # --- Luật 1: DELETE (Xóa) ---
    delete_cond = (
        (df['is_temp_file'] == 1) | 
        ((df['size_bytes'] < DELETE_SIZE_THRESHOLD) & (df['days_since_access'] > DELETE_TIME_THRESHOLD))
    )
    df.loc[delete_cond, 'Label'] = 'Delete'
    
    # --- Luật 2: COMPRESS (Nén/Lưu trữ) ---
    compress_cond = (
        (df['size_bytes'] > COMPRESS_SIZE_THRESHOLD) & 
        (df['days_since_access'] > COMPRESS_TIME_THRESHOLD) & 
        (~df['extension'].isin(COMPRESSED_EXTS)) 
    )
    df.loc[compress_cond & (df['Label'] == 'Keep'), 'Label'] = 'Compress'
    
    return df

# ========================= HÀM CHÍNH (MAIN FUNCTION) =========================
def main():
    
    print("================== TRỢ LÝ DỌN DẸP ĐĨA ML - HUẤN LUYỆN ===================")
    
    # -------------------------------------------------------------
    # PHASE 1: TẠO DỮ LIỆU GIẢ VÀ HUẤN LUYỆN (TRAINING)
    # -------------------------------------------------------------
    
    # BƯỚC 1: TẠO TẬP DỮ LIỆU GIẢ VÀ GÁN NHÃN (GROUND TRUTH)
    synthetic_df_labeled = generate_synthetic_metadata()
    
    print("\n============== PHÂN BỔ NHÃN GIẢ (Training Ground Truth) ==============")
    print(synthetic_df_labeled['Label'].value_counts())

    # BƯỚC 2: HUẤN LUYỆN MÔ HÌNH
    X_synth = synthetic_df_labeled[['size_log', 'days_since_access', 'is_temp_file']]
    y_synth = synthetic_df_labeled['Label']

    le = LabelEncoder()
    y_encoded_synth = le.fit_transform(y_synth)
    
    if len(synthetic_df_labeled) < 20: 
        print("Không đủ dữ liệu giả để huấn luyện mô hình. Thoát chương trình.")
        sys.exit(0)
    
    # Chia tập huấn luyện và kiểm tra mô hình
    X_train, X_test, y_train, y_test = train_test_split(
        X_synth, y_encoded_synth, test_size=0.2, random_state=42)

    print("\n[BƯỚC 3] Bắt đầu huấn luyện mô hình Decision Tree...")
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Độ chính xác mô hình trên tập kiểm tra giả: {accuracy*100:.2f}%")
    
    # -------------------------------------------------------------
    # BƯỚC 4: LƯU MÔ HÌNH
    # -------------------------------------------------------------
    ASSETS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    
    print("\n[BƯỚC 4] Hoàn tất lưu trữ mô hình và bộ mã hóa:")
    print(f"   -> Mô hình được lưu tại: {MODEL_FILE}")
    print(f"   -> Bộ mã hóa được lưu tại: {ENCODER_FILE}")
    print("\nBây giờ bạn có thể chạy disk_cleaner.py để sử dụng mô hình.")

if __name__ == '__main__':
    main()
