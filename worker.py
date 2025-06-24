# worker.py - ФИНАЛЬНАЯ ВЕРСИЯ

import os, time, json, requests, io, traceback, uuid, redis, boto3
import cv2
import numpy as np
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

# Инициализируем клиент Replicate с нашими настройками таймаута
replicate_client = replicate.Client(
    api_token=REPLICATE_API_TOKEN,
    timeout=180.0  # Устанавливаем таймаут 180 секунд для всех запросов
)
# --- КОНЕЦ НОВОГО БЛОКА ---

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
    """Запускает модель через клиент с настроенным таймаутом и надежно опрашивает статус."""
    print(f"-> Запуск модели '{description}'...")
    # Используем наш новый клиент replicate_client
    prediction = replicate_client.predictions.create(version=version, input=input_data)
    print(f"   -> Replicate задача создана с ID: {prediction.id}")

    start_time = time.time()
    timeout = 600

    while True:
        try:
            print(f"   -> Запрос актуального статуса для '{description}' (ID: {prediction.id})...")
            # Используем наш новый клиент replicate_client
            prediction = replicate_client.predictions.get(prediction.id)
            print(f"   -> Статус получен: {prediction.status}")
        except Exception as e:
            print(f"   -> !!! ВНИМАНИЕ: Ошибка при получении статуса для '{description}': {e}. Повторная попытка...")
            time.sleep(5)
            continue

        if prediction.status in ["succeeded", "failed", "canceled"]:
            break

        if time.time() - start_time > timeout:
            raise Exception(f"Модель '{description}' не завершилась за {timeout} секунд (таймаут).")

        time.sleep(3)

    if prediction.status != 'succeeded':
        raise Exception(f"Модель '{description}' не удалась со статусом {prediction.status}. Ошибка: {prediction.error}")

    print(f"<- Модель '{description}' успешно завершена.")
    return prediction.output

def composite_images(original_url, upscaled_url, mask_url):
    print("-> Начало композитинга изображений (метод OpenCV)...")
    try:
        MAX_RESOLUTION = 4096

        def url_to_image(url, flags=cv2.IMREAD_UNCHANGED):
            resp = requests.get(url, stream=True).raw
            image_array = np.asarray(bytearray(resp.read()), dtype="uint8")
            return cv2.imdecode(image_array, flags)

        original_img_bgr = url_to_image(original_url, cv2.IMREAD_COLOR)
        upscaled_img_bgr = url_to_image(upscaled_url, cv2.IMREAD_COLOR)
        mask_img_gray = url_to_image(mask_url, cv2.IMREAD_GRAYSCALE)

        # Конвертируем в BGRA (4 канала)
        original_img = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2BGRA)
        upscaled_img = cv2.cvtColor(upscaled_img_bgr, cv2.COLOR_BGR2BGRA)

        h, w = upscaled_img.shape[:2]
        if max(h, w) > MAX_RESOLUTION:
            scale = MAX_RESOLUTION / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            print(f"   -> Изображение слишком большое. Ограничиваем до {new_w}x{new_h}px.")
            upscaled_img = cv2.resize(upscaled_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        h, w = upscaled_img.shape[:2]
        expand_size = int(w * 0.05)
        expand_size = expand_size if expand_size % 2 != 0 else expand_size + 1

        print(f"   -> Расширяем маску (OpenCV kernel size: {expand_size})...")
        kernel = np.ones((expand_size, expand_size), np.uint8)
        mask_resized = cv2.resize(mask_img_gray, (w, h), interpolation=cv2.INTER_NEAREST)
        expanded_mask = cv2.dilate(mask_resized, kernel, iterations=1)

        blur_size = int(expand_size * 0.2)
        blur_size = blur_size if blur_size % 2 != 0 else blur_size + 1
        print(f"   -> Растушевываем края (OpenCV blur size: {blur_size})...")
        soft_mask = cv2.GaussianBlur(expanded_mask, (blur_size, blur_size), 0)

        soft_mask_float = soft_mask.astype(np.float32) / 255.0

        # --- ИСПРАВЛЕНИЕ ЗДЕСЬ: делаем маску 4-канальной (BGRA), а не 3-канальной ---
        soft_mask_alpha = cv2.cvtColor(soft_mask_float, cv2.COLOR_GRAY2BGRA)

        original_resized = cv2.resize(original_img, (w, h), interpolation=cv2.INTER_AREA)

        # Приводим типы данных к единому для смешивания
        composite = (soft_mask_alpha * upscaled_img.astype(np.float32)) + ((1 - soft_mask_alpha) * original_resized.astype(np.float32))
        composite = composite.astype(np.uint8)

        original_h, original_w = original_img_bgr.shape[:2] # Используем оригинал без альфа-канала для правильного размера
        final_image_bgr = cv2.resize(composite, (original_w, original_h), interpolation=cv2.INTER_AREA)

        # Конвертируем финальное изображение в BGR (3 канала) перед сохранением в PNG
        final_image_to_save = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGRA2BGR)

        _, image_data_encoded = cv2.imencode('.png', final_image_to_save)
        image_data = io.BytesIO(image_data_encoded)
        image_data.seek(0)

        print("<- Композитинг (OpenCV) успешно завершен.")
        return image_data

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Ошибка на этапе композитинга (OpenCV): {e}")

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
            num_inference_steps = 40
        elif original_width <= 4096: # 4K
            scale_factor = 4.0
            creativity = 0.40
            resemblance = 1.20
            hdr = 2 # <--- ИСПРАВЛЕНО
            num_inference_steps = 40
        else: # 6K и больше
            scale_factor = 4.0
            creativity = 0.30
            resemblance = 1.50
            hdr = 1 # <--- ИСПРАВЛЕНО
            num_inference_steps = 40
            
        upscaler_input = {
            "image": generated_image_url,
            "scale_factor": scale_factor,
            "creativity": creativity,
            "resemblance": resemblance,
            "num_inference_steps": num_inference_steps,
            "dynamic": hdr  # Модель использует параметр 'dynamic' для HDR
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
