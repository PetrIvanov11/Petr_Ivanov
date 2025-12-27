FROM python:3.10

# Устанавливаем системные зависимости для LightGBM и других библиотек
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы
COPY . .

# Запускаем решение
CMD ["python", "main.py"]