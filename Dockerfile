FROM python:3.10-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY main.py /app/main.py
ENV PORT=5005

# 暴露端口（可选）
EXPOSE ${PORT}

ENTRYPOINT ["python","main.py"]