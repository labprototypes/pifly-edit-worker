import os
import time
import json
import requests
import io

import replicate
import redis
import boto3
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import sessionmaker

# --- НАСТРОЙКА ---
# Загружаем переменные окружения, они должны быть установлены на Render
DATABASE_URL = os.environ.get('DATABASE_URL')
REDIS_URL = os.environ.get('REDIS_URL')
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')
AWS_S3_REGION = os.environ.get('AWS_S3_REGION')

# --- МОДЕЛИ REPLICATE ---
# Версии моделей, которые мы будем использовать в цепочке
FLUX_MODEL_VERSION = "black-forest-labs/flux-kontext-max:0b9c317b23e79a9a0d8b9602ff4d04030d433055927fb7c4b91c44234a6818c4"
SAM_MODEL_VERSION = "lucataco/segment-anything:50700142f7c65c697816f15779743c36691a03f8f94e9b95764790409a67c446"
UPSCALER_MODEL_VERSION = "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c5fcf05f45d25456d209595473143a84F"

# --- ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ ---
# Создаем минимальное Flask-приложение только для контекста SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Копируем модели из вашего основного app.py, чтобы воркер "знал" структуру таблиц
class User(db.Model):
    id = db.Column(db.String(128), primary_key=True)
    token_balance = db.Column(db.Integer, default=100, nullable=False)
    # ... можно добавить остальные поля, но для воркера достаточно этих

class Prediction(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(128), nullable=False)
    replicate_id = db.Column(db.String(255), unique=True, nullable=True)
    status = db.Column(db.String(50), default='pending', nullable=False)
    output_url = db.Column(db.String(2048), nullable=True)
    token_cost = db.Column(db.Integer, nullable=False, default=1)

# --- ФУНКЦИИ-ПОМОЩНИКИ ---

def run_replicate_model(version, input_data, description):
    """Универсальная функция для запуска модели Replicate и ожидания результата."""
    print(f"-> Запуск модели '{description}'...")
    prediction = replicate.predictions.create(version=version, input=input_data)
    prediction.wait()
    if prediction.status != 'succeeded':
        raise Exception(f"Модель '{description}' не удалась со статусом {prediction.status}. Ошибка: {prediction.error}")
    print(f"<- Модель '{description}' успешно завершена.")
    return prediction.output

def composite_images(original_url, upscaled_url, mask_url):
    """Скачивает три изображения и собирает финальное."""
    print("-> Начало композитинга изображений...")
    try:
        original_img = Image.open(requests.get(original_url, stream=True).raw).convert("RGBA")
        upscaled_img = Image.open(requests.get(upscaled_url, stream=True).raw).convert("RGBA")
        mask_img = Image.open(requests.get(mask_url, stream=True).raw).convert("L")

        upscaled_img = upscaled_img.resize(original_img.size, Image.LANCZOS)

        final_img = Image.composite(upscaled_img, original_img, mask_img)
        
        # Сохраняем в байтовый поток в памяти, чтобы не создавать временный файл
        image_data = io.BytesIO()
        final_img.save(image_data, format='PNG')
        image_data.seek(0)
        
        print("<- Композитинг успешно завершен.")
        return image_data
    except Exception as e:
        raise Exception(f"Ошибка на этапе композитинга: {e}")

def upload_to_s3(image_data, user_id, prediction_id):
    """Загружает финальное изображение в наш S3 бакет."""
    print("-> Загрузка финального результата в S3...")
    s3_client = boto3.client('s3', region_name=AWS_S3_REGION, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    object_name = f"generations/{user_id}/{prediction_id}.png"
    
    s3_client.upload_fileobj(image_data, AWS_S3_BUCKET_NAME, object_name, ExtraArgs={'ContentType': 'image/png'})
    
    permanent_s3_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com/{object_name}"
    print(f"<- Результат сохранен в S3: {permanent_s3_url}")
    return permanent_s3_url

### НАЧАЛО БЛОКА ДЛЯ ЗАМЕНЫ ФУНКЦИИ ###

def process_job(job_data, db_session):
    """Основная логика обработки одной задачи."""
    prediction_id = job_data['prediction_id']
    print(f"--- Начало обработки задачи {prediction_id} ---")
    
    try:
        # Шаг 1: Генерация измененного изображения с помощью FLUX
        flux_output = run_replicate_model(
            FLUX_MODEL_VERSION,
            {"input_image": job_data['original_s3_url'], "prompt": job_data['prompt']},
            "FLUX Edit"
        )
        # ОТЛАДКА: Печатаем результат от FLUX
        print(f"!!! РЕЗУЛЬТАТ FLUX: {flux_output}")
        # ИСПРАВЛЕНИЕ: Некоторые модели возвращают список, поэтому берем первый элемент.
        generated_image_url = flux_output[0] if isinstance(flux_output, list) else flux_output
        print(f"   -> Используем URL для следующих шагов: {generated_image_url}")


        # Шаг 2: Создание маски измененной области с помощью SAM
        mask_output = run_replicate_model(
            SAM_MODEL_VERSION,
            {"image": generated_image_url, "prompt": job_data['prompt']},
            "SAM Masking"
        )
        # ОТЛАДКА: Печатаем результат от SAM
        print(f"!!! РЕЗУЛЬТАТ SAM: {mask_output}")
        # ИСПРАВЛЕНИЕ: Убеждаемся, что получили словарь с ключом 'mask'
        if not isinstance(mask_output, dict) or 'mask' not in mask_output:
            raise Exception(f"Неожиданный формат ответа от модели SAM: {mask_output}")
        mask_url = mask_output['mask']
        print(f"   -> Используем URL маски: {mask_url}")


        # Шаг 3: Апскейл сгенерированного изображения
        upscaled_output = run_replicate_model(
            UPSCALER_MODEL_VERSION,
            {"image": generated_image_url},
            "Upscaler"
        )
        # ОТЛАДКА: Печатаем результат от Upscaler
        print(f"!!! РЕЗУЛЬТАТ UPSCALER: {upscaled_output}")
        # ИСПРАВЛЕНИЕ: Некоторые модели возвращают список, поэтому берем первый элемент.
        upscaled_image_url = upscaled_output[0] if isinstance(upscaled_output, list) else upscaled_output
        print(f"   -> Используем URL апскейла: {upscaled_image_url}")


        # Шаг 4: Композитинг
        final_image_data = composite_images(job_data['original_s3_url'], upscaled_image_url, mask_url)
        
        # Шаг 5: Загрузка результата в наш S3
        final_s3_url = upload_to_s3(final_image_data, job_data['user_id'], prediction_id)

        # Шаг 6: Обновляем запись в БД со статусом 'completed'
        prediction = db_session.query(Prediction).get(prediction_id)
        prediction.status = 'completed'
        prediction.output_url = final_s3_url
        db_session.commit()
        print(f"--- Задача {prediction_id} успешно завершена! ---")

    except Exception as e:
        print(f"!!! ОШИБКА при обработке задачи {prediction_id}: {e}")
        prediction = db_session.query(Prediction).get(prediction_id)
        if prediction:
            prediction.status = 'failed'
            user = db_session.query(User).get(prediction.user_id)
            if user:
                user.token_balance += prediction.token_cost
                print(f"Возвращено {prediction.token_cost} токенов пользователю {user.id}")
            db_session.commit()

### КОНЕЦ БЛОКА ДЛЯ ЗАМЕНЫ ФУНКЦИИ ###

# --- ОСНОВНОЙ ЦИКЛ ВОРКЕРА ---

def main():
    # Эта команда "открывает двери" в офис для нашего воркера
    with app.app_context():
        # Весь остальной код теперь сдвинут вправо (имеет отступ)
        print(">>> Воркер PiflyEdit запущен и ожидает задач...")
        redis_client = redis.from_url(REDIS_URL)
        
        Session = sessionmaker(bind=db.engine)
        
        while True:
            try:
                _, job_json = redis_client.brpop('pifly_edit_jobs')
                job_data = json.loads(job_json)
                
                session = Session()
                process_job(job_data, session)
                session.close()

            except Exception as e:
                print(f"!!! КРИТИЧЕСКАЯ ОШИБКА в основном цикле воркера: {e}")
                time.sleep(5)

if __name__ == "__main__":
    main()
