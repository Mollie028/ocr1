# 使用輕量級的 Python 基底映像
FROM python:3.10-slim

# 安裝 PaddleOCR 所需系統套件（libgl1 + libgomp1 + libgthread）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製依賴檔案並安裝
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 複製應用程式原始碼
COPY . .

# 用 uvicorn 啟動 FastAPI 應用，注意：不要手寫 port 數字
CMD ["python", "-m", "uvicorn", "ocr_api:app", "--host", "0.0.0.0", "--port", "8000"]


