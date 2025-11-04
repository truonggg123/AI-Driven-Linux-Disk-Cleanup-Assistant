import os
import pandas as pd
import numpy as np
import random 
import joblib # Thư viện để lưu và tải mô hình
from pathlib import Path
import sys # Import sys để thoát chương trình

# Thư viện cho Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report 

# ==================== CẤU HÌNH DỮ LIỆU GIẢ VÀ LUẬT GÁN NHÃN ====================
# Thư mục lưu trữ mô hình đã huấn luyện
ASSETS_DIR = Path("ml_assets")
MODEL_FILE = ASSETS_DIR / "disk_model.joblib"
ENCODER_FILE = ASSETS_DIR / "label_encoder.joblib"

# Cấu hình cho việc tạo dữ liệu giả (CHỈ DÙNG CHO TRAINING)
SYNTHETIC_NUM_FILES = 5000 
SYNTHETIC_MAX_SIZE_MB = 200
SYNTHETIC_MAX_AGE_DAYS = 730
SYNTHETIC_EXTENSIONS = [ 
    ".pdf", ".docx", ".xlsx", ".zip", ".rar", ".7z", 
    ".jpg", ".png", ".mp4", ".mov", ".gif", ".iso", 
    ".log", ".out", ".tmp", ".bak", ".cache", ".~",  
    ".py", ".sh", ".c", ".cpp", ".h", ".html", 
    ".txt", ".js", ".json", ".xml", ".db", ""
]

# === LUẬT GÁN NHÃN ĐƯỢC CẢI TIẾN CHO THỰC TẾ HƠN ===
# NGƯỠNG CỦA HÀNH ĐỘNG DELETE/COMPRESS
DELETE_SIZE_THRESHOLD = 5 * 1024 * 1024      # Tệp nhỏ (< 5MB)
DELETE_TIME_THRESHOLD = 120                  # Tệp nhỏ không dùng > 120 ngày (giảm từ 180)

COMPRESS_SIZE_THRESHOLD = 50 * 1024 * 1024 # Tệp lớn (> 50MB)
COMPRESS_TIME_THRESHOLD = 60                 # Tệp lớn không dùng > 60 ngày (giảm từ 90)
MAX_COMPRESS_AGE = 500                       # Tuổi tối đa cho tệp nén. Tệp > 500 ngày sẽ bị xóa.

# CÁC LOẠI TỆP CẦN XỬ LÝ ĐẶC BIỆT
LOG_EXTENSIONS = [".log", ".out"]
LOG_DELETE_SIZE_THRESHOLD = 100 * 1024 * 1024 # Logs lớn (> 100MB) bị ưu tiên xóa.
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

# -------------------- HÀM XỬ LÝ VÀ GÁN NHÃN (Logic Cải tiến) --------------------
def label_data(df):
    """Tính toán Features (đặc trưng) và gán nhãn Ground Truth (Ground Truth)."""
    if df.empty:
        return df

    temp_extensions = [".log", ".tmp", ".bak", ".cache", ".~", ""]
    
    # Kỹ thuật Đặc trưng (Feature Engineering)
    df['is_log_file'] = df['extension'].isin(LOG_EXTENSIONS).astype(int) 
    df['is_temp_file'] = df['extension'].isin(temp_extensions).astype(int) 
    df['size_log'] = np.log10(df['size_bytes'] + 1)
    
    df['Label'] = 'Keep'

    # --- Luật A: DELETE (Log/Cache Rác Lớn) - Ưu tiên Xóa ---
    # Logs lớn (>100MB) bị ưu tiên xóa vì thường là dấu hiệu lỗi.
    log_delete_cond = (
        (df['is_log_file'] == 1) & (df['size_bytes'] > LOG_DELETE_SIZE_THRESHOLD) 
    )
    df.loc[log_delete_cond, 'Label'] = 'Delete'
    
    # --- Luật B: DELETE (Tệp Tạm/Tệp Nhỏ Cũ) ---
    delete_cond = (
        (df['Label'] == 'Keep') & # Chỉ áp dụng cho tệp chưa bị Luật A gắn nhãn
        (
            (df['is_temp_file'] == 1) | 
            ((df['size_bytes'] < DELETE_SIZE_THRESHOLD) & (df['days_since_access'] > DELETE_TIME_THRESHOLD))
        )
    )
    df.loc[delete_cond, 'Label'] = 'Delete'
    
    # --- Luật C: COMPRESS (Tệp Lớn Cũ Vừa) ---
    compress_cond = (
        (df['Label'] == 'Keep') & # Chỉ áp dụng cho tệp chưa bị gắn nhãn
        (df['size_bytes'] > COMPRESS_SIZE_THRESHOLD) & 
        (df['days_since_access'] > COMPRESS_TIME_THRESHOLD) & 
        (df['days_since_access'] < MAX_COMPRESS_AGE) & # Thêm ngưỡng tuổi tối đa
        (~df['extension'].isin(COMPRESSED_EXTS)) 
    )
    df.loc[compress_cond, 'Label'] = 'Compress'
    
    # --- Luật D: DELETE (Tệp Rất Rất Cũ) ---
    # Tệp đã quá cũ (ví dụ: > 500 ngày) sẽ được chuyển thành Xóa thay vì Nén.
    very_old_delete_cond = (df['days_since_access'] >= MAX_COMPRESS_AGE) 
    # Chỉ ghi đè nhãn 'Compress' và 'Keep' (nhãn 'Delete' đã được xử lý ở trên)
    df.loc[very_old_delete_cond & (df['Label'] != 'Delete'), 'Label'] = 'Delete'

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
    # Cải tiến ML: Thêm class_weight='balanced' để xử lý mất cân bằng lớp
    model = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Độ chính xác mô hình trên tập kiểm tra giả: {accuracy*100:.2f}%")
    
    # BÁO CÁO PHÂN LOẠI CHI TIẾT
    y_pred = model.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)
    print("\n================== BÁO CÁO PHÂN LOẠI (Precision/Recall) ==================")
    print(classification_report(y_test_labels, y_pred_labels))
    print("==========================================================================")
    
    # -------------------------------------------------------------
    # BƯỚC 4: LƯU MÔ HÌNH
    # -------------------------------------------------------------
    ASSETS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    
    print("\n[BƯỚC 4] Hoàn tất lưu trữ mô hình và bộ mã hóa:")
    print(f"    -> Mô hình được lưu tại: {MODEL_FILE}")
    print(f"    -> Bộ mã hóa được lưu tại: {ENCODER_FILE}")
    print("\nBây giờ bạn có thể chạy disk_cleaner.py để sử dụng mô hình.")

if __name__ == '__main__':
    main()
