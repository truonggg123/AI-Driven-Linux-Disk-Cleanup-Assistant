import os
import time
import shutil
import sys
from pathlib import Path
import platform 

# ==================== CẤU HÌNH KIỂM THỬ (PHẢI KHỚP VỚI MÔ HÌNH) ====================
SANDBOX_DIR = Path("ml_test_sandbox")
CURRENT_TIME = time.time() # Thời gian hiện tại tính bằng giây (epoch)

# Ngưỡng được lấy từ train_model.py
DELETE_TIME_THRESHOLD = 180 # days
COMPRESS_TIME_THRESHOLD = 90 # days

# Mục đích của hàm này là tạo tệp và thao tác thời gian truy cập (atime)
def create_and_touch_file(filepath: Path, size_mb: float, days_ago: int, reason: str):
    """Tạo một tệp với kích thước và thời gian truy cập (atime) mong muốn."""
    try:
        size_bytes = int(size_mb * 1024 * 1024)
        
        # 1. Ghi nội dung để tạo tệp với kích thước gần đúng
        with open(filepath, 'wb') as f:
            f.write(os.urandom(size_bytes)) 

        # 2. Tính toán thời gian truy cập (atime) và thay đổi (mtime) mục tiêu
        target_time = CURRENT_TIME - (days_ago * 24 * 3600)
        
        # 3. Sử dụng os.utime để thay đổi thời gian truy cập và thay đổi
        # st_atime (thời gian truy cập) là yếu tố quyết định trong mô hình của bạn
        os.utime(filepath, (target_time, target_time))
        
        print(f"   [OK] {filepath.name:<25} -> {reason:<25} ({size_mb:.1f} MB, {days_ago} ngày)")
        
    except Exception as e:
        print(f"   [LỖI] Không thể tạo tệp {filepath.name}: {e}")

def main():
    print("================== TẠO MÔI TRƯỜNG KIỂM NGHIỆM ĐĨA TOÀN DIỆN ==================")
    
    # Dọn dẹp môi trường cũ
    if SANDBOX_DIR.exists():
        print(f"Đang xóa thư mục kiểm nghiệm cũ: {SANDBOX_DIR.resolve()}")
        try:
            shutil.rmtree(SANDBOX_DIR)
        except Exception as e:
            print(f"Không thể xóa thư mục cũ. Vui lòng xóa thủ công. Lỗi: {e}")
            sys.exit(1)
            
    # 1. Tạo thư mục sandbox mới
    SANDBOX_DIR.mkdir(exist_ok=True)
    print(f"[BƯỚC 1] Đã tạo thư mục kiểm nghiệm tại: {SANDBOX_DIR.resolve()}")
    
    print("\n[BƯỚC 2] Tạo 9 tệp giả lập (3 Delete, 3 Compress, 3 Keep):")

    # ==============================================================================
    # 1. TỆP ĐỀ XUẤT XÓA (DELETE) - Luật: (Junk) HOẶC (Size < 5MB AND Age > 180 ngày)
    # ==============================================================================
    print("\n--- A. TỆP ĐỀ XUẤT XÓA (EXPECTED: DELETE) ---")
    
    # 1.1 Xóa theo phần mở rộng (Junk file, bất kể kích thước/tuổi)
    create_and_touch_file(
        SANDBOX_DIR / "junk_log_recent.log", 
        size_mb=10.0, 
        days_ago=10, 
        reason="Junk File (log)"
    )

    # 1.2 Xóa theo luật Size/Age (Nhỏ & Rất cũ)
    create_and_touch_file(
        SANDBOX_DIR / "old_small_doc.txt", 
        size_mb=3.0, 
        days_ago=250, 
        reason="Small (<5MB) & Old (>180d)"
    )

    # 1.3 Xóa theo luật Size/Age (Nhỏ & Cực cũ)
    create_and_touch_file(
        SANDBOX_DIR / "tiny_cache.cache", 
        size_mb=0.1, 
        days_ago=500, 
        reason="Junk File (cache)"
    )

    # ==============================================================================
    # 2. TỆP ĐỀ XUẤT NÉN (COMPRESS) - Luật: (Size > 50MB AND Age > 90 ngày)
    # ==============================================================================
    print("\n--- B. TỆP ĐỀ XUẤT NÉN (EXPECTED: COMPRESS) ---")

    # 2.1 Lớn và khá cũ (ngay trên ngưỡng 90 ngày)
    create_and_touch_file(
        SANDBOX_DIR / "large_video_old.mp4", 
        size_mb=100.0, 
        days_ago=150, 
        reason="Large (>50MB) & Old (>90d)"
    )

    # 2.2 Rất lớn và rất cũ
    create_and_touch_file(
        SANDBOX_DIR / "huge_report.pdf", 
        size_mb=75.0, 
        days_ago=300, 
        reason="Large (>50MB) & Very Old"
    )

    # 2.3 Lớn, mới chạm ngưỡng tuổi
    create_and_touch_file(
        SANDBOX_DIR / "mega_data.csv", 
        size_mb=120.0, 
        days_ago=100, 
        reason="Large (>50MB) & Near 90d"
    )
    
    # ==============================================================================
    # 3. TỆP ĐỀ XUẤT GIỮ LẠI (KEEP) - Luật: Không thỏa mãn Delete/Compress
    # ==============================================================================
    print("\n--- C. TỆP ĐỀ XUẤT GIỮ LẠI (EXPECTED: KEEP) ---")

    # 3.1 Lớn nhưng RẤT MỚI (Fail Compress Age: < 90 ngày)
    create_and_touch_file(
        SANDBOX_DIR / "new_large_project.iso", 
        size_mb=90.0, 
        days_ago=30, 
        reason="Large (>50MB) but Recent (<90d)"
    )

    # 3.2 Nhỏ và Cũ, nhưng là file nén (Không bị Compress, không bị Delete)
    create_and_touch_file(
        SANDBOX_DIR / "small_compressed.zip", 
        size_mb=1.0, 
        days_ago=400, 
        reason="Small & Compressed (Skip)"
    )

    # 3.3 Nhỏ và MỚI (Fail Delete Age: < 180 ngày)
    create_and_touch_file(
        SANDBOX_DIR / "recent_code_file.py", 
        size_mb=2.0, 
        days_ago=50, 
        reason="Small (<5MB) & Recent (<180d)"
    )


    print("\n[HOÀN TẤT] Môi trường kiểm nghiệm đã sẵn sàng với 9 tệp có chủ đích.")
    print(f"Tiếp theo, hãy chạy 'python disk_cleaner.py' và nhập đường dẫn: {SANDBOX_DIR.resolve()}")

if __name__ == '__main__':
    main()
