from flask import Flask, request, jsonify
from flask_cors import CORS # Để xử lý vấn đề CORS khi ReactJS gọi API
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import io # Để làm việc với dữ liệu ảnh từ request
from PIL import Image # Pillow để đọc ảnh từ bytes

app = Flask(__name__)
CORS(app) # Cho phép tất cả các domain (trong môi trường dev, có thể cấu hình chặt hơn cho production)

# --- CẤU HÌNH VÀ TẢI MODEL ---
MODEL_PATH = 'TrafficSign_Improved_Best_Model.keras' # Đặt file model cùng thư mục hoặc cung cấp đường dẫn đúng
SIGNNAMES_CSV_PATH = 'signnames.csv' # Đặt file csv cùng thư mục hoặc cung cấp đường dẫn đúng

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model đã được tải thành công từ: {MODEL_PATH}")
except Exception as e:
    print(f"Lỗi khi tải model: {e}")
    model = None # Xử lý lỗi nếu không tải được model

try:
    data_df = pd.read_csv(SIGNNAMES_CSV_PATH)
    class_id_to_name_en = pd.Series(data_df.SignName.values, index=data_df.ClassId).to_dict()
    print("Signnames CSV đã được tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải signnames.csv: {e}")
    class_id_to_name_en = {}

# Dictionary dịch tên (bạn cần hoàn thiện nó)
sign_name_vn = {
    "Speed limit (20km/h)": "Giới hạn tốc độ (20km/h)",  # ClassId 0 - Có lỗi chính tả "imit" trong file CSV gốc
    "Speed limit (30km/h)": "Giới hạn tốc độ (30km/h)",  # ClassId 1
    "Speed limit (50km/h)": "Giới hạn tốc độ (50km/h)",  # ClassId 2
    "Speed limit (60km/h)": "Giới hạn tốc độ (60km/h)",  # ClassId 3
    "Speed limit (70km/h)": "Giới hạn tốc độ (70km/h)",  # ClassId 4
    "Speed limit (80km/h)": "Giới hạn tốc độ (80km/h)",  # ClassId 5
    "End of speed limit (80km/h)": "Hết giới hạn tốc độ (80km/h)",  # ClassId 6
    "Speed limit (100km/h)": "Giới hạn tốc độ (100km/h)", # ClassId 7
    "Speed limit (120km/h)": "Giới hạn tốc độ (120km/h)", # ClassId 8
    "No passing": "Cấm vượt",  # ClassId 9
    "No passing for vechiles over 3.5 metric tons": "Cấm vượt cho xe trên 3.5 tấn",  # ClassId 10 - Có lỗi chính tả "vechiles"
    "Right-of-way at the next intersection": "Ưu tiên qua nơi giao nhau tiếp theo",  # ClassId 11
    "Priority road": "Đường ưu tiên",  # ClassId 12
    "Yield": "Nhường đường",  # ClassId 13
    "Stop": "Dừng lại",  # ClassId 14
    "No vechiles": "Cấm các loại xe",  # ClassId 15 - Có lỗi chính tả "vechiles"
    "Vechiles over 3.5 metric tons prohibited": "Cấm xe trên 3.5 tấn",  # ClassId 16 - Có lỗi chính tả "Vechiles"
    "No entry": "Cấm vào",  # ClassId 17
    "General caution": "Cảnh báo nguy hiểm chung",  # ClassId 18
    "Dangerous curve to the left": "Chỗ ngoặt nguy hiểm vòng bên trái",  # ClassId 19
    "Dangerous curve to the right": "Chỗ ngoặt nguy hiểm vòng bên phải",  # ClassId 20
    "Double curve": "Nhiều chỗ ngoặt nguy hiểm liên tiếp",  # ClassId 21 (Thường là cua trái rồi cua phải hoặc ngược lại)
    "Bumpy road": "Đường gồ ghề",  # ClassId 22
    "Slippery road": "Đường trơn trượt",  # ClassId 23
    "Road narrows on the right": "Đường bị thu hẹp về bên phải",  # ClassId 24
    "Road work": "Công trường",  # ClassId 25
    "Traffic signals": "Đèn tín hiệu giao thông",  # ClassId 26
    "Pedestrians": "Người đi bộ cắt ngang",  # ClassId 27
    "Children crossing": "Trẻ em qua đường",  # ClassId 28
    "Bicycles crossing": "Xe đạp cắt ngang",  # ClassId 29
    "Beware of ice/snow": "Cẩn thận đường đóng băng/tuyết",  # ClassId 30
    "Wild animals crossing": "Thú rừng vượt qua đường",  # ClassId 31
    "End of all speed and passing limits": "Hết mọi lệnh cấm (tốc độ và vượt)",  # ClassId 32
    "Turn right ahead": "Hướng phải đi vòng sang phải",  # ClassId 33
    "Turn left ahead": "Hướng phải đi vòng sang trái",  # ClassId 34
    "Ahead only": "Chỉ được đi thẳng",  # ClassId 35
    "Go straight or right": "Đi thẳng hoặc rẽ phải",  # ClassId 36
    "Go straight or left": "Đi thẳng hoặc rẽ trái",  # ClassId 37
    "Keep right": "Đi về bên phải",  # ClassId 38
    "Keep left": "Đi về bên trái",  # ClassId 39
    "Roundabout mandatory": "Nơi giao nhau chạy theo vòng xuyến",  # ClassId 40
    "End of no passing": "Hết cấm vượt",  # ClassId 41
    "End of no passing by vechiles over 3.5 metric tons": "Hết cấm vượt cho xe trên 3.5 tấn"  # ClassId 42 - Có lỗi chính tả "vechiles"
}
# --- HÀM PREPROCESSING (Sao chép từ notebook Colab) ---
def preprocessing_for_api(img_cv2):
    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    img_laplacian_float = cv2.Laplacian(img_gray, cv2.CV_64F)
    img_normalized = img_laplacian_float / 255.0
    return img_normalized

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model không khả dụng.'}), 500
    if not class_id_to_name_en:
        return jsonify({'error': 'Dữ liệu tên biển báo không khả dụng.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Không có file nào được gửi lên.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file nào.'}), 400

    try:
        # Đọc ảnh từ memory
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Chuyển PIL Image sang OpenCV format (BGR)
        img_cv2_original = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Tiền xử lý ảnh
        img_resized = cv2.resize(img_cv2_original, (32, 32))
        img_preprocessed = preprocessing_for_api(img_resized)
        img_reshaped = img_preprocessed.reshape(1, 32, 32, 1)

        # Dự đoán
        prediction_probs = model.predict(img_reshaped, verbose=0)
        predicted_class_id = int(np.argmax(prediction_probs)) # Chuyển sang int để jsonify
        confidence = float(np.max(prediction_probs)) # Chuyển sang float

        # Lấy tên biển báo
        sign_name_en = class_id_to_name_en.get(predicted_class_id, "Không rõ (ID không tồn tại)")
        sign_name_display_vn = sign_name_vn.get(sign_name_en, sign_name_en)

        return jsonify({
            'predicted_class_id': predicted_class_id,
            'sign_name_en': sign_name_en,
            'sign_name_vn': sign_name_display_vn,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # Chạy server