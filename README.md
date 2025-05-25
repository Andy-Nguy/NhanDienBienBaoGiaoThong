# NhanDienBienBaoGiaoThong bằng mô hình CNN

Đây là một ứng dụng(giao diện web đơn giản xây dựng bằng ReactJS) cho phép người dùng tải lên hình ảnh biển báo giao thông và nhận dạng loại biển báo đó bằng mô hình Deep Learning (Convolutional Neural Network - CNN)

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Cấu trúc Project](#cấu-trúc-project)
- [Yêu cầu Cài đặt](#yêu-cầu-cài-đặt)
- [Hướng dẫn Chạy Ứng dụng](#hướng-dẫn-chạy-ứng-dụng)
  - [Chạy bằng Docker (Khuyến nghị)](#chạy-bằng-docker-khuyến-nghị)
  - [Chạy Thủ công (Không dùng Docker)](#chạy-thủ-công-không-dùng-docker)
- [Cách sử dụng](#cách-sử-dụng)
- [Lưu ý về File Model](#lưu-ý-về-file-model)
- [Các Bước Phát triển Tiếp theo (Tùy chọn)](#các-bước-phát-triển-tiếp-theo-tùy-chọn)

## Giới thiệu

Ứng dụng này được xây dựng với mục đích nhận diện các loại biển báo giao thông từ hình ảnh đầu vào. Backend sử dụng mô hình CNN được huấn luyện trên bộ dữ liệu biển báo giao thông (ví dụ: German Traffic Sign Recognition Benchmark - GTSRB) để phân loại. Frontend cung cấp giao diện người dùng thân thiện để tải ảnh và xem kết quả.

## Công nghệ sử dụng

*   **Backend:**
    *   Python
    *   Flask (Web framework)
    *   TensorFlow / Keras (Deep Learning framework để xây dựng và chạy mô hình CNN)
    *   OpenCV (Xử lý ảnh)
    *   NumPy, Pandas (Thao tác dữ liệu)
*   **Frontend:**
    *   ReactJS (Thư viện JavaScript để xây dựng giao diện người dùng)
    *   Axios (Thư viện HTTP client để gọi API)
    *   HTML, CSS
*   **Đóng gói & Triển khai (Tùy chọn):**
    *   Docker & Docker Compose

## Cấu trúc Project
BE_TEST/
├── backend_flask/
│ ├── TrafficSign_Improved_Best_Model.keras # File model CNN đã huấn luyện
│ ├── app.py # File chính của ứng dụng Flask
│ ├── signnames.csv # File mapping ClassId sang tên biển báo
│ ├── requirements.txt # Các gói Python cần thiết cho backend
│ ├── Dockerfile # Dockerfile cho backend
│ └── venv_tf/ # (Nên nằm trong .gitignore) Thư mục môi trường ảo
├── frontend_react_new/
│ ├── public/ # Các file tĩnh cho React
│ ├── src/ # Mã nguồn React (App.js, App.css, index.js, ...)
│ ├── package.json # Thông tin project và dependencies của Node.js
│ ├── Dockerfile # Dockerfile cho frontend
│ └── node_modules/ # (Nên nằm trong .gitignore)
├── .gitignore # Các file và thư mục Git sẽ bỏ qua
├── docker-compose.yml # File cấu hình Docker Compose
└── README.md # File mô tả này

## Yêu cầu Cài đặt

### Cho Chạy bằng Docker (Khuyến nghị):
1.  **Docker Desktop:** Tải và cài đặt từ [Docker Hub](https://www.docker.com/products/docker-desktop/).
2.  **Git:** (Tùy chọn, để clone repository).

### Cho Chạy Thủ công:
1.  **Python:** Phiên bản 3.9+ (ví dụ: 3.10, 3.11). Đảm bảo `python` và `pip` đã được thêm vào PATH.
2.  **Node.js và npm:** Node.js phiên bản LTS (ví dụ: 18.x hoặc 20.x). Đảm bảo `node` và `npm` đã được thêm vào PATH.
3.  **Git:** (Tùy chọn, để clone repository).

## Hướng dẫn Chạy Ứng dụng

### Chạy bằng Docker (Khuyến nghị)

1.  **Clone Repository:**
    ```bash
    git clone  https://github.com/Andy-Nguy/NhanDienBienBaoGiaoThong.git
    cd BE_TEST
    ```

2.  **(Quan trọng - Nếu file model không được commit):**
    *   Nếu file `TrafficSign_Improved_Best_Model.keras` không có sẵn trong `backend_flask/` sau khi clone (do quá lớn và đã được thêm vào `.gitignore`), bạn cần tải nó về thủ công và đặt vào thư mục `BE_TEST/backend_flask/`.

3.  **Build và Chạy các Container:**
    Từ thư mục gốc của project (nơi chứa `docker-compose.yml`), chạy lệnh:
    ```bash
    docker compose up --build
    ```
    *   Lần đầu tiên chạy, quá trình build có thể mất vài phút.
    *   Các lần sau, nếu không có thay đổi trong `Dockerfile` hoặc code ảnh hưởng đến build, bạn có thể chạy `docker compose up`.

4.  **Truy cập Ứng dụng:**
    *   Mở trình duyệt web và truy cập: `http://localhost:3000`

5.  **Dừng Ứng dụng:**
    *   Nhấn `Ctrl + C` trong cửa sổ terminal đang chạy `docker compose up`.
    *   Để dừng và xóa các container: `docker compose down`

### Chạy Thủ công (Không dùng Docker)

Bạn sẽ cần mở hai cửa sổ terminal riêng biệt.

**Terminal 1: Chạy Backend (Flask)**

1.  Di chuyển vào thư mục backend:
    ```bash
    cd BE_TEST/backend_flask
    ```
2.  Tạo và kích hoạt môi trường ảo:
    ```bash
    python -m venv venv_manual (# Hoặc tên bạn muốn)
    # Windows:
    .\venv_manual\Scripts\activate
    # macOS/Linux:
    # source venv_manual/bin/activate
    ```
3.  Cài đặt các gói phụ thuộc:
    ```bash
    pip install -r requirements.txt
    ```
    *(Đảm bảo bạn đã tạo file `requirements.txt` cho backend từ môi trường phát triển của mình bằng `pip freeze > requirements.txt` và commit nó).*
4.  **(Quan trọng - Nếu file model không được commit):** Đảm bảo file `TrafficSign_Improved_Best_Model.keras` và `signnames.csv` có trong thư mục `backend_flask/`.
5.  Chạy server Flask:
    ```bash
    python app.py
    ```
    Server sẽ chạy ở `http://localhost:5000`.

**Terminal 2: Chạy Frontend (ReactJS)**

1.  Di chuyển vào thư mục frontend:
    ```bash
    cd BE_TEST/frontend_react_new
    ```
2.  Cài đặt các gói phụ thuộc:
    ```bash
    npm install
    ```
3.  Chạy ứng dụng React:
    ```bash
    npm start
    ```
    Ứng dụng sẽ mở trong trình duyệt ở `http://localhost:3000`.

## Cách sử dụng

1.  Mở trình duyệt và truy cập vào địa chỉ của Frontend (ví dụ: `http://localhost:3000` nếu chạy cục bộ).
2.  Nhấn nút "Chọn Ảnh" để tải lên một hình ảnh biển báo giao thông từ máy tính của bạn (hỗ trợ các định dạng .png, .jpg, .jpeg, kích thước tối đa 5MB).
3.  Ảnh xem trước sẽ hiển thị.
4.  Nhấn nút "Bắt đầu Nhận dạng".
5.  Chờ một lát để ứng dụng xử lý và hiển thị kết quả dự đoán, bao gồm tên biển báo (tiếng Việt và tiếng Anh), Class ID, và độ tin cậy của dự đoán.
6.  Bạn có thể chọn "Xóa ảnh" để thử với một ảnh khác.

## Lưu ý về File Model

*   File model CNN (`TrafficSign_Improved_Best_Model.keras`) được sử dụng có kích thước khoảng [48.6MB].
*   Nếu bạn không commit file này lên GitHub do giới hạn kích thước, hãy đảm bảo người dùng tải về và đặt nó vào thư mục `backend_flask/` trước khi chạy ứng dụng (đặc biệt là trước khi build Docker image cho backend).

## Các Bước Phát triển Tiếp theo 

*   Cải thiện độ chính xác của mô hình bằng cách thử nghiệm các kiến trúc, kỹ thuật tiền xử lý, hoặc data augmentation khác.
*   Phân tích lỗi chi tiết hơn để hiểu các trường hợp model dự đoán sai.
*   Tối ưu hóa tốc độ dự đoán của model (ví dụ: TensorFlow Lite).
*   Cải thiện giao diện người dùng và trải nghiệm người dùng.
*   Thêm tính năng cho phép người dùng phản hồi về kết quả dự đoán.
*   Triển khai ứng dụng lên một nền tảng cloud.

---

Trong quá trình phát triển thì Project vẫn còn một số điểm chưa hoàn thiện,mong thầy có thể xem qua và góp ý để nhóm ghi nhận và hoàn thiện sản phẩm , hiểu bài tốt hơn




#Một số lệnh 
1.python -m venv venv_tf_cloned (# Đặt tên khác nếu muốn, ví dụ venv )
  . .\venv_tf_cloned\Scripts\activate

2.pip install Flask tensorflow opencv-python pandas numpy Pillow

3.pip install Flask-Cors

  File TrafficSign_Improved_Best_Model.keras đang cho kết quả train đúng nhất
