FROM python:3.13-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir . && chmod +x start.sh

CMD ["bash", "start.sh"]
