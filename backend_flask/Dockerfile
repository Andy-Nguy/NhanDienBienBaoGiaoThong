# Sử dụng base image Python chính thức
FROM python:3.10-slim   
# Bạn có thể chọn phiên bản Python phù hợp (ví dụ 3.9, 3.11)

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Thêm các thư viện khác nếu OpenCV vẫn báo lỗi phụ thuộc
    && rm -rf /var/lib/apt/lists/* # Xóa cache để giảm kích thước image

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào thư mục làm việc
COPY requirements.txt .

# Cài đặt các thư viện Python cần thiết
# --no-cache-dir để giảm kích thước image
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ nội dung của thư mục backend_flask cục bộ vào /app trong container
COPY . .

# Chỉ định port mà ứng dụng Flask sẽ chạy bên trong container
EXPOSE 5000

# Lệnh để chạy ứng dụng Flask khi container khởi động
# Sử dụng Gunicorn cho production thay vì Flask development server
# RUN pip install gunicorn # Cài Gunicorn
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# Hoặc dùng Flask development server (đơn giản hơn cho dev, nhưng không khuyến khích cho production)
CMD ["python", "app.py"]