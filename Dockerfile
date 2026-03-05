FROM python:3.12.12

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 再复制应用代码
COPY . .

EXPOSE 5010
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5010"]