### НАЧАЛО: КОД ДЛЯ ПОЛНОЙ ЗАМЕНЫ ФАЙЛА WORKER.PY ###

import os
import time
import json
import requests
import io
import traceback
import uuid

import replicate
import redis
import boto3
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import sessionmaker
from PIL import Image

# --- НАСТРОЙКА ---
DATABASE_URL = os.environ.get('DATABASE_URL')
REDIS_URL = os.environ.get('REDIS_URL')
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')
AWS_S3_REGION = os.environ.get('AWS_S3_REGION')

# --- МОДЕЛИ REPLICATE ---
FLUX_MODEL_VERSION = "black-forest-labs/flux-kontext-max:0b9c317b23e79a9a0d8b9602ff4d04030d433055927fb7c4b91c44234a6818c4"
# Мы все еще пытаемся использовать правильную версию SAM
SAM_MODEL_VERSION = "tmappdev/lang-segment-anything:46424b33633644367f035f29d7249911e3b5e91a033526f8d7441a7e4683a45c"
UPSCALER_MODEL_VERSION = "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c5fcf05f45d25456d209595473143a84F"

# --- ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- МОДЕЛИ БД ---
class User(db.Model):
    id = db.Column(db.String(128), primary_key=True)
    token_balance = db.Column(db.Integer, nullable=False)

class Prediction(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    output_url = db.Column(db.String(2048), nullable=True)
    token_cost = db.Column(db.Integer, nullable=False)

# --- ФУНКЦИИ-ПОМОЩНИКИ ---
def composite_images(original_url, upscaled_url, mask_url):
    print("-> Начало композитинга изображений...")
    try:
        original_img = Image.open(requests.get(original_url, stream=True).raw).convert("RGBA")
        upscaled_img = Image.open(requests.get(upscaled_url, stream=True).raw).convert("RGBA")
        mask_img = Image.open(requests.get(mask_url, stream=True).raw).convert("L")
        upscaled_img = upscaled_img.resize(original_img.size, Image.LANCZOS)
        final_img = Image.composite(upscaled_img, original_img, mask_img)
        image_data = io.BytesIO()
        final_img.save(image_data, format='PNG')
        image_data.seek(0)
        print("<- Композитинг успешно завершен.")
        return image_data
    except Exception as e:
        raise Exception(f"Ошибка на этапе композитинга: {e}")

def run_replicate_model(version, input_data, description):
    print(f"-> Запуск модели '{description}'...")
    prediction = replicate.predictions.create(version=version, input=input_data)
    prediction.wait()
    if prediction.status != 'succeeded':
        raise Exception(f"Модель '{description}' не удалась со статусом {prediction.status}. Ошибка: {prediction.error}")
    print(f"<- Модель '{description}' успешно завершена.")
    return prediction.output

def composite_images(original_url, upscaled_url, mask_url):
    # ... (эта функция остается без изменений) ...
    print("-> Начало композитинга изображений...")
    try:
        original_img = Image.open(requests.get(original_url, stream=True).raw).convert("RGBA")
        upscaled_img = Image.open(requests.get(upscaled_url, stream=True).raw).convert("RGBA")
        mask_img = Image.open(requests.get(mask_url, stream=True).raw).convert("L")
        upscaled_img = upscaled_img.resize(original_img.size, Image.LANCZOS)
        final_img = Image.composite(upscaled_img, original_img, mask_img)
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
    
    # Генерируем уникальное имя для финального файла
    object_name = f"generations/{user_id}/{prediction_id}-final.png"
    
    # Загружаем данные из памяти в S3
    s3_client.upload_fileobj(
        image_data,
        AWS_S3_BUCKET_NAME,
        object_name,
        ExtraArgs={'ContentType': 'image/png'}
    )
    
    permanent_s3_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com/{object_name}"
    print(f"<- Финальный результат сохранен в S3: {permanent_s3_url}")
    return permanent_s3_url

# --- ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ ---
# --- ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ (ИСПРАВЛЕНА И УПРОЩЕНА) ---
def process_job(job_data, db_session):
    prediction_id = job_data['prediction_id']
    print(f"--- Начало обработки задачи {prediction_id} ---")
    try:
        # Шаг 1: Генерация FLUX
        flux_output = run_replicate_model(FLUX_MODEL_VERSION, {"input_image": job_data['original_s3_url'], "prompt": job_data['prompt']}, "FLUX Edit")
        generated_image_url = flux_output[0] if isinstance(flux_output, list) else flux_output

        # Шаг 2: Создание маски (используем правильную модель и правильный параметр)
        sam_input = {"image": generated_image_url, "text_prompt": job_data['prompt']}
        mask_output = run_replicate_model(SAM_MODEL_VERSION, sam_input, "Lang-SAM Masking")
        mask_url = mask_output[0] if isinstance(mask_output, list) else mask_output

        # Шаг 3: Апскейл
        upscaled_output = run_replicate_model(UPSCALER_MODEL_VERSION, {"image": generated_image_url}, "Upscaler")
        upscaled_image_url = upscaled_output[0] if isinstance(upscaled_output, list) else upscaled_output

        # Шаг 4: Композитинг
        final_image_data = composite_images(job_data['original_s3_url'], upscaled_image_url, mask_url)

        # Шаг 5: Загрузка финального результата в S3
        final_s3_url = upload_to_s3(final_image_data, job_data['user_id'], prediction_id)

        # Шаг 6: Обновляем запись в БД
        prediction = db_session.query(Prediction).get(prediction_id)
        if prediction:
            prediction.status = 'completed'
            prediction.output_url = final_s3_url
            db_session.commit()
            print(f"--- ПОЛНАЯ ЗАДАЧА {prediction_id} УСПЕШНО ЗАВЕРШЕНА! ---")

    except Exception as e:
        print(f"!!! ОШИБКА при обработке задачи {prediction_id}:")
        traceback.print_exc()
        prediction = db_session.query(Prediction).get(prediction_id)
        if prediction:
            prediction.status = 'failed'
            user = db_session.query(User).get(prediction.user_id)
            if user:
                user.token_balance += prediction.token_cost
                print(f"Возвращено {prediction.token_cost} токенов пользователю {user.id}")
            db_session.commit()

# --- ОСНОВНОЙ ЦИКЛ ВОРКЕРА ---
def main():
    with app.app_context():
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

### КОНЕЦ: КОД ДЛЯ ПОЛНОЙ ЗАМЕНЫ ФАЙЛА WORKER.PY ###
