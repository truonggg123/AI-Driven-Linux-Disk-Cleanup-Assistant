#!/bin/bash
# Script cài đặt PyTorch CPU-only trên Linux (không cần NVIDIA driver/CUDA)

echo "=========================================="
echo "Cài đặt PyTorch CPU-only cho Kali Linux"
echo "=========================================="
echo ""
echo "Lưu ý: Phiên bản này KHÔNG cần NVIDIA driver hay CUDA"
echo ""

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "Lỗi: Không tìm thấy Python3"
    exit 1
fi

echo "Đang cài đặt PyTorch CPU-only..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "=========================================="
echo "Kiểm tra cài đặt..."
echo "=========================================="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Hoàn tất! PyTorch đã được cài đặt ở chế độ CPU-only."
echo "Bạn có thể chạy ứng dụng mà không cần NVIDIA GPU."

