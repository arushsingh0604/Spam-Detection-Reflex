FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (IMPORTANT: unzip is required by Reflex)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

# Initialize Reflex (now unzip is available)
RUN reflex init

EXPOSE 3000

CMD ["reflex", "run", "--env", "prod", "--host", "0.0.0.0"]
