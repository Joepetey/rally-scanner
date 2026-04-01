FROM python:3.13-slim

WORKDIR /app

# Install rally-ml workspace package first (better layer caching)
COPY packages/rally-ml /app/packages/rally-ml
RUN pip install --no-cache-dir /app/packages/rally-ml

# Install the main application
COPY . .
RUN pip install --no-cache-dir . && chmod +x start.sh

CMD ["bash", "start.sh"]
