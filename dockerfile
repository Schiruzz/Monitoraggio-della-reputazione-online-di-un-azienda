FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
EXPOSE 7860
CMD ["python", "app.py"]