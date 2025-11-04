import os
import pandas as pd
import numpy as np
import random 
import sys
import joblib # Thư viện để lưu và tải mô hình
from pathlib import Path

# Thư viện cho Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import spacy

# ==================== CẤU HÌNH DỮ LIỆU GIẢ VÀ LUẬT GÁN NHÃN ====================
# Thư mục lưu trữ mô hình đã huấn luyện
# Khởi tạo spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
    print("[INFO] Đã tải mô hình spaCy: en_core_web_lg")
except OSError:
    print("[LỖI] Không tìm thấy mô hình spaCy 'en_core_web_lg'.")
    print("Vui lòng chạy: python -m spacy download en_core_web_lg")
    sys.exit(1)

ASSETS_DIR = Path("ml_assets")
MODEL_FILE = ASSETS_DIR / "disk_model.joblib"
ENCODER_FILE = ASSETS_DIR / "label_encoder.joblib"
PCA_FILE = ASSETS_DIR / "pca_transformer.joblib"  # Lưu PCA transformer cho embedding vectors

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

# Từ khóa để tạo tên file có ý nghĩa hơn
SYNTHETIC_KEYWORDS = {
    'temp': ['temp', 'temporary', 'tmp', 'cache', 'old', 'backup'],
    'important': ['document', 'report', 'project', 'important', 'final'],
    'media': ['photo', 'image', 'video', 'movie', 'picture'],
    'archive': ['archive', 'old', 'backup', 'old_version']
}

# Các ngưỡng này định nghĩa Ground Truth cho mô hình
DELETE_SIZE_THRESHOLD = 5 * 1024 * 1024   # 5MB
DELETE_TIME_THRESHOLD = 180               # 180 ngày

COMPRESS_SIZE_THRESHOLD = 50 * 1024 * 1024 # 50MB
COMPRESS_TIME_THRESHOLD = 90              # 90 ngày
COMPRESSED_EXTS = [".zip", ".rar", ".7z", ".tar.gz", ".gz"]
# ==============================================================================


# ----------------- HÀM TRÍCH XUẤT FEATURES TỪ TÊN FILE BẰNG SPACY -----------------
def extract_spacy_features(file_path):
    """
    Trích xuất features từ tên file sử dụng spaCy NLP.
    Trả về: dict chứa các features từ NLP
    """
    # Lấy tên file không có extension
    file_name = Path(file_path).stem.lower()
    
    # Xử lý bằng spaCy
    doc = nlp(file_name)
    
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
    
    # Feature 5: Embedding vector từ tên file (giảm chiều bằng PCA sau)
    # Sử dụng vector trung bình của các token
    if len(doc) > 0 and doc.vector is not None:
        embedding_vector = doc.vector  # Vector 300D từ en_core_web_lg
    else:
        embedding_vector = np.zeros(300)  # Vector 0 nếu không có token
    
    return {
        'num_words': num_words,
        'name_length': name_length,
        'has_temp_keyword': int(has_temp_keyword),
        'has_important_keyword': int(has_important_keyword),
        'embedding_vector': embedding_vector
    }

# ----------------- HÀM TẠO METADATA GIẢ TRONG BỘ NHỚ -----------------
def generate_synthetic_metadata():
    """Tạo metadata giả trong bộ nhớ để huấn luyện mô hình."""
    file_data_list = []
    
    print(f"\n[BƯỚC 1] Bắt đầu tạo {SYNTHETIC_NUM_FILES} metadata giả trong bộ nhớ...")

    # Tạo tên file có ý nghĩa hơn
    file_prefixes = ['document', 'report', 'photo', 'video', 'temp_file', 'cache', 
                     'backup', 'old_file', 'project', 'data', 'log', 'archive']
    
    for i in range(SYNTHETIC_NUM_FILES):
        ext = random.choice(SYNTHETIC_EXTENSIONS)
        # Kích thước ngẫu nhiên theo log scale
        size_bytes = int(10 ** (random.uniform(2, np.log10(SYNTHETIC_MAX_SIZE_MB * 1024 * 1024)))) 
        days_ago = random.randint(1, SYNTHETIC_MAX_AGE_DAYS)
        
        # Tạo tên file có ý nghĩa hơn
        prefix = random.choice(file_prefixes)
        file_name = f"{prefix}_{i:04d}{ext}"
        
        file_data_list.append({
            'file_path': file_name,
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
    
    # Trích xuất features từ tên file bằng spaCy
    print("   Đang trích xuất features từ tên file bằng spaCy...")
    spacy_features = []
    embedding_vectors = []
    
    for idx, file_path in enumerate(df['file_path']):
        features = extract_spacy_features(file_path)
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
    
    # Lưu embedding vectors để giảm chiều sau
    df['_embedding_vector'] = embedding_vectors
    
    df['Label'] = 'Keep'

    # --- Luật 1: DELETE (Xóa) ---
    # Cập nhật điều kiện để bao gồm các từ khóa temp trong tên file
    delete_cond = (
        (df['is_temp_file'] == 1) | 
        (df['has_temp_keyword'] == 1) |  # Thêm điều kiện mới từ spaCy
        ((df['size_bytes'] < DELETE_SIZE_THRESHOLD) & (df['days_since_access'] > DELETE_TIME_THRESHOLD))
    )
    df.loc[delete_cond, 'Label'] = 'Delete'
    
    # --- Luật 2: COMPRESS (Nén/Lưu trữ) ---
    compress_cond = (
        (df['size_bytes'] > COMPRESS_SIZE_THRESHOLD) & 
        (df['days_since_access'] > COMPRESS_TIME_THRESHOLD) & 
        (~df['extension'].isin(COMPRESSED_EXTS)) &
        (df['has_important_keyword'] == 0)  # Không nén các file quan trọng
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

    # BƯỚC 2: CHUẨN BỊ DỮ LIỆU CHO HUẤN LUYỆN
    print("\n[BƯỚC 2] Chuẩn bị features cho huấn luyện...")
    
    # Lấy embedding vectors và giảm chiều bằng PCA
    embedding_matrix = np.array(list(synthetic_df_labeled['_embedding_vector']))
    print(f"   Kích thước embedding vectors: {embedding_matrix.shape}")
    
    # Sử dụng PCA để giảm từ 300D xuống 10D
    pca = PCA(n_components=10, random_state=42)
    embedding_reduced = pca.fit_transform(embedding_matrix)
    
    # Tạo tên cột cho embedding features
    embedding_cols = [f'embedding_dim_{i}' for i in range(10)]
    embedding_df = pd.DataFrame(embedding_reduced, columns=embedding_cols, index=synthetic_df_labeled.index)
    
    # Kết hợp tất cả features
    basic_features = synthetic_df_labeled[['size_log', 'days_since_access', 'is_temp_file',
                                           'num_words', 'name_length', 'has_temp_keyword', 'has_important_keyword']]
    X_synth = pd.concat([basic_features, embedding_df], axis=1)
    
    print(f"   Tổng số features: {X_synth.shape[1]}")
    print(f"   Features: {list(X_synth.columns)}")
    
    y_synth = synthetic_df_labeled['Label']

    le = LabelEncoder()
    y_encoded_synth = le.fit_transform(y_synth)
    
    if len(synthetic_df_labeled) < 20: 
        print("Không đủ dữ liệu giả để huấn luyện mô hình. Thoát chương trình.")
        sys.exit(0)
    
    # Chia tập huấn luyện và kiểm tra mô hình
    X_train, X_test, y_train, y_test = train_test_split(
        X_synth, y_encoded_synth, test_size=0.2, random_state=42)

    print("\n[BƯỚC 3] Bắt đầu huấn luyện mô hình Decision Tree với features từ spaCy...")
    model = DecisionTreeClassifier(max_depth=8, random_state=42)  # Tăng depth để xử lý nhiều features hơn
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Độ chính xác mô hình trên tập kiểm tra giả: {accuracy*100:.2f}%")
    
    # -------------------------------------------------------------
    # BƯỚC 4: LƯU MÔ HÌNH VÀ CÁC TRANSFORMERS
    # -------------------------------------------------------------
    ASSETS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    joblib.dump(pca, PCA_FILE)  # Lưu PCA transformer để sử dụng khi dự đoán
    
    print("\n[BƯỚC 4] Hoàn tất lưu trữ mô hình và các transformers:")
    print(f"   -> Mô hình được lưu tại: {MODEL_FILE}")
    print(f"   -> Bộ mã hóa được lưu tại: {ENCODER_FILE}")
    print(f"   -> PCA transformer được lưu tại: {PCA_FILE}")
    print("\nBây giờ bạn có thể chạy disk_cleaner.py để sử dụng mô hình.")

if __name__ == '__main__':
    main()
