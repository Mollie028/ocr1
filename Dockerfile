FROM python:3.9-slim-buster

# 如果軟體包不在預設軟體倉庫中，請添加任何必要的軟體倉庫
# RUN echo "deb http://archive.ubuntu.com/ubuntu focal main universe" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    # 其他依賴項

# 複製您的應用程式碼
COPY . /app

# 安裝 Python 依賴項
WORKDIR /app
RUN pip install -r requirements.txt

# 定義您的進入點
CMD ["python", "your_app.py"]
