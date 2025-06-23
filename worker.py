# worker.py - ФИНАЛЬНАЯ ВЕРСИЯ

import os, time, json, requests, io, traceback, uuid, redis, boto3
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import sessionmaker
from PIL import Image, ImageFilter
import replicate

DATABASE_URL = os.environ.get('DATABASE_URL')
REDIS_URL = os.environ.get('REDIS_URL')
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')
AWS_S3_REGION = os.environ.get('AWS_S3_REGION')

# --- МОДЕЛИ REPLICATE (ФИНАЛЬНЫЕ ВЕРСИИ, КОТОРЫЕ ВЫ ВЫБРАЛИ) ---
FLUX_MODEL_VERSION = "black-forest-labs/flux-kontext-max:0b9c317b23e79a9a0d8b9602ff4d04030d433055927fb7c4b91c44234a6818c4"
SAM_MODEL_VERSION = "tmappdev/lang-segment-anything:891411c38a6ed2d44c004b7b9e44217df7a5b07848f29ddefd2e28bc7cbf93bc"
UPSCALER_MODEL_VERSION = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.String(128), primary_key=True)
    token_balance = db.Column(db.Integer, nullable=False)

class Prediction(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    output_url = db.Column(db.String(2048), nullable=True)
    token_cost = db.Column(db.Integer, nullable=False)

# --- ВСТАВЬТЕ ЭТОТ КОД НА МЕСТО СТАРОЙ ФУНКЦИИ run_replicate_model ---

def run_replicate_model(version, input_data, description):
    """Запускает модель и активно опрашивает ее статус вместо пассивного ожидания."""
    print(f"-> Запуск модели '{description}'...")
    
    # Создаем предсказание
    prediction = replicate.predictions.create(version=version, input=input_data)
    print(f"   -> Replicate задача создана с ID: {prediction.id}")

    start_time = time.time()
    # Устанавливаем максимальное время ожидания, например, 5 минут (300 секунд)
    timeout = 300 

    while prediction.status not in ["succeeded", "failed", "canceled"]:
        # Проверяем, не вышли ли мы за таймаут
        if time.time() - start_time > timeout:
            raise Exception(f"Модель '{description}' не завершилась за {timeout} секунд (таймаут).")

        # Ждем 2 секунды перед следующим запросом
        time.sleep(2)
        # Обновляем статус предсказания
        prediction.reload()
        print(f"   -> Проверка статуса для '{description}': {prediction.status}")

    if prediction.status != 'succeeded':
        raise Exception(f"Модель '{description}' не удалась со статусом {prediction.status}. Ошибка: {prediction.error}")

    print(f"<- Модель '{description}' успешно завершена.")
    return prediction.output

# --- ВСТАВЬТЕ ЭТОТ КОД НА МЕСТО СТАРОЙ ФУНКЦИИ composite_images ---

# --- ВСТАВЬТЕ ЭТОТ КОД НА МЕСТО СТАРОЙ ФУНКЦИИ composite_images ---

# --- ВСТАВЬТЕ ЭТОТ КОД НА МЕСТО СТАРОЙ ФУНКЦИИ composite_images ---

def composite_images(original_url, upscaled_url, mask_url):
    print("-> Начало композитинга изображений (финальная логика)...")
    try:
        # Загружаем оригинал и маску. Они должны быть одного размера.
        original_img = Image.open(requests.get(original_url, stream=True).raw).convert("RGBA")
        mask_img = Image.open(requests.get(mask_url, stream=True).raw).convert("L")

        # Загружаем увеличенную версию ИЗМЕНЕННОГО ОБЪЕКТА
        upscaled_patch = Image.open(requests.get(upscaled_url, stream=True).raw).convert("RGBA")

        # --- НОВАЯ, ПРАВИЛЬНАЯ ЛОГИКА ---
        # 1. Уменьшаем наш высококачественный патч до размера оригинала.
        #    За счет этого он сохраняет больше деталей, чем если бы мы не делали апскейл.
        final_res_patch = upscaled_patch.resize(original_img.size, Image.LANCZOS)

        # 2. Обрабатываем маску, как и договаривались (расширение + растушевка)
        expand_size = int(mask_img.width * 0.10)
        expand_size = expand_size if expand_size % 2 != 0 else expand_size + 1
        expanded_mask = mask_img.filter(ImageFilter.MaxFilter(size=expand_size))
        
        blur_radius = int(expand_size * 1.0)
        soft_mask = expanded_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 3. Теперь "наклеиваем" качественный патч нужного размера на ОРИГИНАЛ по мягкой маске
        # Фон остается нетронутым и сохраняет 100% исходного качества.
        final_image = Image.composite(final_res_patch, original_img, soft_mask)

        image_data = io.BytesIO()
        final_image.save(image_data, format='PNG')
        image_data.seek(0)
        
        print("<- Композитинг успешно завершен.")
        return image_data
    except Exception as e:
        raise Exception(f"Ошибка на этапе композитинга: {e}")

def upload_to_s3(image_data, user_id, prediction_id):
    print("-> Загрузка финального результата в S3...")
    s3_client = boto3.client('s3', region_name=AWS_S3_REGION, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    object_name = f"generations/{user_id}/{prediction_id}-final.png"
    s3_client.upload_fileobj(image_data, AWS_S3_BUCKET_NAME, object_name, ExtraArgs={'ContentType': 'image/png'})
    permanent_s3_url = f"https://{AWS_S3_BUCKET_NAME}.s3.{AWS_S3_REGION}.amazonaws.com/{object_name}"
    print(f"<- Финальный результат сохранен в S3: {permanent_s3_url}")
    return permanent_s3_url

def process_job(job_data, db_session):
    """Финальная версия с умным апскейлом"""
    prediction_id = job_data['prediction_id']
    print(f"--- Начало обработки задачи {prediction_id} ---")
    try:
        # Шаг 1: Генерация FLUX (без изменений)
        flux_output = run_replicate_model(FLUX_MODEL_VERSION, {"input_image": job_data['original_s3_url'], "prompt": job_data['generation_prompt']}, "FLUX Edit")
        generated_image_url = flux_output[0] if isinstance(flux_output, list) else flux_output

        # Шаг 2: Создание маски (без изменений)
        intent = job_data.get('intent')
        image_for_masking = generated_image_url if intent == 'ADD' else job_data['original_s3_url']
        sam_input = {"image": image_for_masking, "text_prompt": job_data['mask_prompt']}
        mask_output = run_replicate_model(SAM_MODEL_VERSION, sam_input, "Lang-SAM Masking")
        mask_url = mask_output[0] if isinstance(mask_output, list) else mask_output
        
        # --- НОВАЯ ЛОГИКА АПСКЕЙЛА ---
        # Шаг 3: Определяем параметры апскейла на основе размера исходного изображения
        original_width = job_data.get('original_width', 1024) # 1024 - значение по умолчанию
        
        if original_width <= 2048: # 2K и меньше
            scale_factor = 2.0
            creativity = 0.40
            resemblance = 1.20
            hdr = 2 # <--- ИСПРАВЛЕНО
        elif original_width <= 4096: # 4K
            scale_factor = 4.0
            creativity = 0.40
            resemblance = 1.20
            hdr = 2 # <--- ИСПРАВЛЕНО
        else: # 6K и больше
            scale_factor = 6.0
            creativity = 0.30
            resemblance = 1.50
            hdr = 1 # <--- ИСПРАВЛЕНО
            
        upscaler_input = {
            "image": generated_image_url,
            "scale_factor": scale_factor,
            "creativity": creativity,
            "resemblance": resemblance,
            "dynamic": hdr # Модель использует параметр 'dynamic' для HDR
        }
        
        upscaled_output = run_replicate_model(UPSCALER_MODEL_VERSION, upscaler_input, "Upscaler")
        upscaled_image_url = upscaled_output[0] if isinstance(upscaled_output, list) else upscaled_output
        
        # Шаги 4, 5, 6 (Композитинг, Сохранение, Обновление БД) без изменений
        final_image_data = composite_images(job_data['original_s3_url'], upscaled_image_url, mask_url)
        final_s3_url = upload_to_s3(final_image_data, job_data['user_id'], prediction_id)

        prediction = db.session.get(Prediction, prediction_id)
        if prediction:
            prediction.status = 'completed'
            prediction.output_url = final_s3_url
            db.session.commit()
            print(f"--- ПОЛНАЯ ЗАДАЧА {prediction_id} УСПЕШНО ЗАВЕРШЕНА! ---")

    except Exception as e:
        print(f"!!! ОШИБКА при обработке задачи {prediction_id}:")
        traceback.print_exc()
        prediction = db_session.get(Prediction, prediction_id)
        if prediction:
            prediction.status = 'failed'
            user = db_session.get(User, prediction.user_id)
            if user:
                user.token_balance += prediction.token_cost
                print(f"Возвращено {prediction.token_cost} токенов пользователю {user.id}")
            db_session.commit()

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
