# --- Giai đoạn 1: Build React App ---
FROM node:18-alpine AS builder 
 # Sử dụng Node.js image (chọn phiên bản LTS, alpine cho kích thước nhỏ)

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép package.json và package-lock.json (hoặc yarn.lock)
COPY package*.json ./
# COPY yarn.lock ./ # Nếu bạn dùng yarn

# Cài đặt dependencies
RUN npm install
# RUN yarn install # Nếu bạn dùng yarn

# Sao chép toàn bộ code source của frontend vào /app
COPY . .

# Build ứng dụng React thành các file tĩnh
RUN npm run build
# RUN yarn build # Nếu bạn dùng yarn

# --- Giai đoạn 2: Phục vụ các file tĩnh bằng Nginx ---
FROM nginx:stable-alpine

# Sao chép các file build tĩnh từ giai đoạn 'builder' vào thư mục phục vụ của Nginx
COPY --from=builder /app/build /usr/share/nginx/html

# (Tùy chọn) Sao chép file cấu hình Nginx tùy chỉnh nếu cần
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Chỉ định port mà Nginx sẽ lắng nghe (mặc định là 80)
EXPOSE 80

# Lệnh để khởi động Nginx
CMD ["nginx", "-g", "daemon off;"]