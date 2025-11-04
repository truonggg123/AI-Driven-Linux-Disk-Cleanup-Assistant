import os
import time
import pandas as pd
import numpy as np
import zipfile 
import sys
import joblib # Thư viện để tải mô hình đã huấn luyện
from pathlib import Path
import spacy

# ==================== CẤU HÌNH VÀ VỊ TRÍ LƯU TRỮ MÔ HÌNH ====================
# Thư mục lưu trữ mô hình đã huấn luyện (Được tạo bởi train_model.py)
ASSETS_DIR = Path("ml_assets")
MODEL_FILE = ASSETS_DIR / "disk_model.joblib"
ENCODER_FILE = ASSETS_DIR / "label_encoder.joblib"
PCA_FILE = ASSETS_DIR / "pca_transformer.joblib"  # PCA transformer cho embedding vectors

# Các ngưỡng này được dùng để tính toán Features (đặc trưng) trên dữ liệu thật.
# Chúng cần phải nhất quán với các ngưỡng đã dùng trong train_model.py
DELETE_SIZE_THRESHOLD = 5 * 1024 * 1024   # 5MB
DELETE_TIME_THRESHOLD = 180               # 180 ngày

COMPRESS_SIZE_THRESHOLD = 50 * 1024 * 1024 # 50MB
COMPRESS_TIME_THRESHOLD = 90              # 90 ngày
COMPRESSED_EXTS = [".zip", ".rar", ".7z", ".tar.gz", ".gz"]
# ==============================================================================

# ----------------- HÀM TRÍCH XUẤT FEATURES TỪ TÊN FILE BẰNG SPACY -----------------
def extract_spacy_features(file_path, nlp_model):
    """
    Trích xuất features từ tên file sử dụng spaCy NLP.
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
    
    # Feature 5: Embedding vector từ tên file (giảm chiều bằng PCA sau)
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

# ----------------- HÀM THU THẬP METADATA THẬT -----------------
def collect_real_metadata(target_dir_path):
    """Quét thư mục thực, thu thập metadata của các tệp (trừ thư mục)."""
    
    current_time = time.time()
    file_data_list = []
    total_files = 0
    
    print(f"\n[BƯỚC 2] Bắt đầu quét thư mục thực: {target_dir_path.resolve()}")
    
    # Sử dụng rglob để quét đệ quy (bao gồm các thư mục con)
    for item_path in target_dir_path.rglob('*'):
        if item_path.is_file():
            total_files += 1
            try:
                stat_info = item_path.stat()
                
                if item_path.is_symlink():
                    continue

                size_bytes = stat_info.st_size
                # st_atime: thời gian truy cập cuối cùng (last access time)
                days_since_access = (current_time - stat_info.st_atime) / (24 * 3600)
                
                # Bỏ qua các tệp quá mới (< 7 ngày)
                if days_since_access < 7: continue

                file_data_list.append({
                    'file_path': item_path.as_posix(),
                    'size_bytes': size_bytes,
                    'extension': item_path.suffix.lower(),
                    'days_since_access': days_since_access,
                })

            except Exception as e:
                print(f"Lỗi khi xử lý {item_path}: {e}")
                continue

    print(f"Hoàn tất quét. Thu thập được {len(file_data_list)}/{total_files} tệp hợp lệ.")
    return pd.DataFrame(file_data_list)

# -------------------- HÀM CHỈ TÍNH TOÁN FEATURE --------------------
def calculate_features(df, nlp_model, pca_transformer):
    """
    Tính toán Features (đặc trưng) cần thiết cho mô hình.
    Bao gồm các features từ spaCy giống như trong train_model.py
    """
    if df.empty:
        return df

    # Đặc trưng 1: Là file tạm thời/rác hay không
    temp_extensions = [".log", ".tmp", ".bak", ".cache", ".~", ""]
    df['is_temp_file'] = df['extension'].isin(temp_extensions).astype(int)

    # Đặc trưng 2: Kích thước tệp (dùng log scale như khi train)
    df['size_log'] = np.log10(df['size_bytes'] + 1)
    
    # Đặc trưng 3: Thời gian kể từ lần truy cập cuối cùng (days_since_access)
    
    # Trích xuất features từ tên file bằng spaCy
    print("   Đang trích xuất features từ tên file bằng spaCy...")
    spacy_features = []
    embedding_vectors = []
    
    for idx, file_path in enumerate(df['file_path']):
        features = extract_spacy_features(file_path, nlp_model)
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
    embedding_reduced = pca_transformer.transform(embedding_matrix)
    
    # Tạo tên cột cho embedding features
    embedding_cols = [f'embedding_dim_{i}' for i in range(embedding_reduced.shape[1])]
    embedding_df = pd.DataFrame(embedding_reduced, columns=embedding_cols, index=df.index)
    
    # Kết hợp tất cả features
    df = pd.concat([df, embedding_df], axis=1)
    
    return df

# ----------------- HÀM ĐỊNH DẠNG KÍCH THƯỚC -----------------
def format_size(size_bytes):
    """Chuyển đổi kích thước tệp từ bytes sang định dạng dễ đọc (KB, MB, GB)."""
    if size_bytes >= (1024**3):
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= (1024**2):
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / 1024:.2f} KB"

# ----------------- HÀM TƯƠNG TÁC VÀ THỰC THI HÀNH ĐỘNG -----------------
def confirm_and_act(suggestions_df, target_dir):
    """Hỏi người dùng và thực hiện hành động THỰC TẾ."""
    
    if suggestions_df.empty:
        return

    # Lấy nhãn dự đoán cho hành động hiện tại (Delete hoặc Compress)
    action_type = suggestions_df['Predicted_Label'].iloc[0]
    
    print(f"\nBạn có muốn thực hiện hành động '{action_type}' trên {len(suggestions_df)} tệp này không?")
    
    # Hiển thị thông tin tệp cho người dùng
    display_cols = ['file_path', 'Formatted_Size', 'days_since_access']
    print(suggestions_df[display_cols].to_string(index=False))
    
    response = input("Nhập 'y' để xác nhận thực hiện hoặc bất kỳ phím nào khác để bỏ qua: ").lower()
    
    if response == 'y':
        print(f"\n--- Thực hiện {action_type} THỰC TẾ ---")
        for index, row in suggestions_df.iterrows():
            file_path = Path(row['file_path'])
            
            if action_type == 'Delete':
                try:
                    file_path.unlink() # THỰC HIỆN XÓA TỆP THẬT SỰ
                    print(f"   [XÓA THỰC TẾ] Đã xóa: {file_path.name}")
                except Exception as e:
                    print(f"   [LỖI XÓA] Không thể xóa {file_path.name}: {e}")

            elif action_type == 'Compress':
                # Tạo thư mục Archive
                archive_dir = target_dir / "ARCHIVE_ML_ASSISTANT"
                archive_dir.mkdir(exist_ok=True)
                
                zip_path = archive_dir / f"{file_path.stem}.zip"
                
                try:
                    # THỰC HIỆN NÉN TỆP THẬT SỰ
                    print(f"   [NÉN THỰC TẾ] Đang nén {file_path.name}...")
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(file_path, file_path.name) 
                    
                    # Xóa tệp gốc sau khi nén thành công
                    file_path.unlink()
                    print(f"   [HOÀN TẤT] Đã nén và xóa gốc: {file_path.name}")
                    
                except Exception as e:
                    print(f"   [LỖI NÉN] Không thể nén {file_path.name}: {e}")

        print(f"Hoàn tất hành động '{action_type}'.")
    else:
        print("Hủy bỏ hành động.")


# ========================= HÀM CHÍNH (MAIN FUNCTION) =========================
def main():
    
    print("================== TRỢ LÝ DỌN DẸP ĐĨA ML - ỨNG DỤNG ===================")
    
    # -------------------------------------------------------------
    # BƯỚC 0: TẢI MÔ HÌNH, BỘ MÃ HÓA, PCA VÀ SPACY MODEL
    # -------------------------------------------------------------
    try:
        model = joblib.load(MODEL_FILE)
        le = joblib.load(ENCODER_FILE)
        pca = joblib.load(PCA_FILE)
        print(f"[BƯỚC 0] Đã tải mô hình từ: {MODEL_FILE}")
        print(f"[BƯỚC 0] Đã tải PCA transformer từ: {PCA_FILE}")
    except FileNotFoundError as e:
        print("\n[LỖI QUAN TRỌNG] Không tìm thấy file mô hình hoặc transformer!")
        print(f"Vui lòng chạy file 'train_model.py' trước để tạo các file cần thiết.")
        print(f"Chi tiết lỗi: {e}")
        sys.exit(1)
    
    # Khởi tạo spaCy model
    try:
        nlp = spacy.load("en_core_web_lg")
        print("[BƯỚC 0] Đã tải mô hình spaCy: en_core_web_lg")
    except OSError:
        print("[LỖI] Không tìm thấy mô hình spaCy 'en_core_web_lg'.")
        print("Vui lòng chạy: python -m spacy download en_core_web_lg")
        sys.exit(1)
    
    # -------------------------------------------------------------
    # BƯỚC 1: XÁC ĐỊNH THƯ MỤC THẬT TẾ
    # -------------------------------------------------------------
    while True:
        target_path_str = input("\n[BƯỚC 1] Nhập đường dẫn thư mục CẦN DỌN DẸP (Ví dụ: /home/user/Downloads): ")
        target_dir = Path(target_path_str)
        if target_dir.is_dir():
            break
        else:
            print("Đường dẫn không hợp lệ hoặc không phải là thư mục. Vui lòng thử lại.")
            
    # -------------------------------------------------------------
    # BƯỚC 2: THU THẬP DỮ LIỆU THẬT & TÍNH TOÁN FEATURE
    # -------------------------------------------------------------
    real_metadata_df = collect_real_metadata(target_dir)

    if real_metadata_df.empty:
        print("Không tìm thấy tệp nào đủ điều kiện để phân tích. Thoát chương trình.")
        sys.exit(0) 
        
    real_df = calculate_features(real_metadata_df, nlp, pca) 
    
    # -------------------------------------------------------------
    # BƯỚC 3: DỰ ĐOÁN & BÁO CÁO
    # -------------------------------------------------------------
    print("\n[BƯỚC 3] Áp dụng mô hình đã huấn luyện để dự đoán hành động trên dữ liệu thật...")
    
    # Chọn các cột feature mà mô hình đã được huấn luyện (giống như khi train)
    feature_cols = ['size_log', 'days_since_access', 'is_temp_file',
                    'num_words', 'name_length', 'has_temp_keyword', 'has_important_keyword']
    # Thêm các embedding dimensions
    embedding_cols = [f'embedding_dim_{i}' for i in range(10)]
    feature_cols.extend(embedding_cols)
    
    # Chỉ lấy các cột có trong DataFrame (tránh lỗi nếu thiếu)
    available_cols = [col for col in feature_cols if col in real_df.columns]
    X_real = real_df[available_cols]
    
    # Dự đoán
    all_predictions_encoded = model.predict(X_real)
    real_df['Predicted_Label'] = le.inverse_transform(all_predictions_encoded)
    
    # Phân loại kết quả
    action_df = real_df[real_df['Predicted_Label'] != 'Keep'].copy()
    action_df['Formatted_Size'] = action_df['size_bytes'].apply(format_size)

    delete_suggestions = action_df[action_df['Predicted_Label'] == 'Delete'] \
        .sort_values(by='size_bytes', ascending=False)
    compress_suggestions = action_df[action_df['Predicted_Label'] == 'Compress'] \
        .sort_values(by='size_bytes', ascending=False)

    print("\n=============== BÁO CÁO ĐỀ XUẤT DỌN DẸP ===============")
    
    # Báo cáo Xóa
    print("\n--- 🗑️ Đề Xuất XÓA (Delete) ---")
    if not delete_suggestions.empty:
        print(f"Tổng số tệp đề xuất xóa: {len(delete_suggestions)}")
        print("Danh sách TOP 5 tệp cần xóa (theo kích thước):")
        print(delete_suggestions.head(5).to_string(columns=['file_path', 'Formatted_Size', 'days_since_access'], index=False))
        confirm_and_act(delete_suggestions.head(5), target_dir) 
    else:
        print("Không có tệp nào được đề xuất xóa.")

    # Báo cáo Nén
    print("\n--- 📦 Đề Xuất NÉN/LƯU TRỮ (Compress) ---")
    if not compress_suggestions.empty:
        print(f"Tổng số tệp đề xuất nén: {len(compress_suggestions)}")
        print("Danh sách TOP 5 tệp cần nén:")
        print(compress_suggestions.head(5).to_string(columns=['file_path', 'Formatted_Size', 'days_since_access'], index=False))
        confirm_and_act(compress_suggestions.head(5), target_dir)
    else:
        print("Không có tệp nào được đề xuất nén.")

    print("\n================== CHƯƠNG TRÌNH KẾT THÚC ==================")

if __name__ == '__main__':
    main()
