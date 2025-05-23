import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css'; // Đảm bảo bạn đã có file App.css

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.size > 5 * 1024 * 1024) { // Giới hạn 5MB
        setErrorMessage('Kích thước file quá lớn (tối đa 5MB).');
        setSelectedFile(null);
        setPreviewImage(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        return;
      }
      setSelectedFile(file);
      setPreviewImage(URL.createObjectURL(file));
      setPredictionResult(null);
      setErrorMessage('');
    } else {
      setSelectedFile(null);
      setPreviewImage(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) {
      setErrorMessage('Vui lòng chọn một file ảnh để nhận dạng.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    setIsLoading(true);
    setErrorMessage('');
    setPredictionResult(null);

    try {
      // --- ĐIỂM QUAN TRỌNG: URL CỦA BACKEND API ---
      const backendApiUrl = 'http://localhost:5000/predict'; // Hoặc IP nếu backend ở máy khác
      // ----------------------------------------------

      const response = await axios.post(backendApiUrl, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // Tăng timeout lên 60 giây (60000ms) nếu dự đoán mất thời gian
      });
      setPredictionResult(response.data);
    } catch (error) {
      console.error("Lỗi khi gửi request dự đoán:", error);
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        setErrorMessage('Yêu cầu vượt quá thời gian chờ. Server có thể đang xử lý hoặc mạng chậm.');
      } else if (error.response) {
        setErrorMessage(`Lỗi từ server (${error.response.status}): ${error.response.data.error || 'Không có thông tin lỗi cụ thể.'}`);
      } else if (error.request) {
        setErrorMessage('Không nhận được phản hồi từ server. Vui lòng kiểm tra kết nối và đảm bảo backend đang chạy.');
      } else {
        setErrorMessage(`Đã xảy ra lỗi khi gửi request: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewImage(null);
    setPredictionResult(null);
    setErrorMessage('');
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="container">
      <header>
        <h1>Ứng dụng Nhận dạng Biển báo Giao thông</h1>
      </header>

      <main>
        <div className="upload-section">
          <input
            type="file"
            accept="image/png, image/jpeg, image/jpg"
            onChange={handleFileChange}
            ref={fileInputRef}
            id="fileInput"
            style={{ display: 'none' }}
          />
          <label htmlFor="fileInput" className="upload-button">
            Chọn Ảnh (Tối đa 5MB)
          </label>

          {selectedFile && (
            <button onClick={handleClear} className="clear-button">
              Xóa ảnh
            </button>
          )}
        </div>

        {errorMessage && (
          <p className="error-message">{errorMessage}</p>
        )}

        {previewImage && (
          <div className="image-preview-container">
            <h3>Xem trước ảnh:</h3>
            <img src={previewImage} alt="Ảnh xem trước" className="preview-image" />
          
          </div>
          
        )}

         {selectedFile && !isLoading && (
      <button
        onClick={handleSubmit}
        disabled={isLoading} // Vẫn giữ disabled khi loading (dù không cần thiết nếu chỉ hiện khi !isLoading)
        className="submit-button main-submit-button" // Thêm class mới để có thể style riêng nếu cần
      >
        Bắt đầu Nhận dạng
      </button>
    )}

        {isLoading && (
          <div className="loading-indicator">
            <p>Đang xử lý, vui lòng chờ...</p>
            {/* Bạn có thể thêm một spinner GIF hoặc CSS ở đây */}
          </div>
        )}

        {predictionResult && !isLoading && ( // Chỉ hiển thị kết quả khi không loading
          <div className="prediction-results">
            <h2>Kết quả Dự đoán:</h2>
            <div className="result-item">
              <span className="result-label">Biển báo (Tiếng Việt):</span>
              <span className="result-value">{predictionResult.sign_name_vn}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Biển báo (Tiếng Anh):</span>
              <span className="result-value">{predictionResult.sign_name_en}</span>
            </div>
            {/* <div className="result-item">
              <span className="result-label">Class ID:</span>
              <span className="result-value">{predictionResult.predicted_class_id}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Độ tin cậy:</span>
              <span className="result-value">
                {(predictionResult.confidence * 100).toFixed(2)}%
              </span>
            </div> */}
          </div>
        )}
      </main>

      <footer>
        <p>© {new Date().getFullYear()} - Ứng dụng Demo</p>
      </footer>
    </div>
  );
}

export default App;