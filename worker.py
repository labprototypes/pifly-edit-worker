# worker.py - ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ

import os, time, json, requests, io, traceback, uuid, redis, boto3
import cv2
import numpy as np
from PIL import Image, ImageFilter
import replicate
import requests
import traceback

# Импорты для автономной работы с БД
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Загрузка переменных окружения
DATABASE_URL = os.environ.get('DATABASE_URL')
REDIS_URL = os.environ.get('REDIS_URL')
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME')
AWS_S3_REGION = os.environ.get('AWS_S3_REGION')
WORKER_SECRET_KEY = os.environ.get('WORKER_SECRET_KEY')

# Модели Replicate
FLUX_MODEL_VERSION = "black-forest-labs/flux-kontext-max:0b9c317b23e79a9a0d8b9602ff4d04030d433055927fb7c4b91c44234a6818c4"
SAM_MODEL_VERSION = "tmappdev/lang-segment-anything:891411c38a6ed2d44c004b7b9e44217df7a5b07848f29ddefd2e28bc7cbf93bc"
UPSCALER_MODEL_VERSION = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN, timeout=180.0)

# Создание независимого подключения к БД
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=280,
    connect_args={"connect_timeout": 60} # <--- ВОТ ЭТО ГЛАВНОЕ ИЗМЕНЕНИЕ
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Определение моделей БД прямо в файле для автономности
class User(Base):
    __tablename__ = 'user'
    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=False)
    email_confirmed = Column(String(1), nullable=False, default='0') # Используем String для совместимости
    email_confirmed_at = Column(String(100), nullable=True) # Используем String для совместимости
    yandex_id = Column(String(255), unique=True, nullable=True)
    token_balance = Column(Integer, default=100, nullable=False)
    marketing_consent = Column(String(1), nullable=False, default='1') # Используем String для совместимости
    subscription_status = Column(String(50), default='free', nullable=False)
    stripe_customer_id = Column(String(255), nullable=True, unique=True)
    stripe_subscription_id = Column(String(255), nullable=True, unique=True)
    current_plan = Column(String(50), nullable=True, default='free')
    trial_used = Column(String(1), default='0', nullable=False) # Используем String для совместимости
    subscription_ends_at = Column(String(100), nullable=True) # Используем String для совместимости


class Prediction(Base):
    __tablename__ = 'prediction'
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), nullable=False) # ИСПРАВЛЕНО: было String(128)
    replicate_id = Column(String(255), unique=True, nullable=True, index=True)
    status = Column(String(50), nullable=False)
    output_url = Column(String(2048), nullable=True)
    created_at = Column(String(100), nullable=True) # Используем String для совместимости
    token_cost = Column(Integer, nullable=False, default=1)

def get_db_session():
    """Создает и возвращает новую сессию БД."""
    return SessionLocal()

# Ваши рабочие функции (без изменений в логике, только для контекста)
def run_replicate_model(version, input_data, description):
    print(f"-> Запуск модели '{description}'...")
    prediction = replicate_client.predictions.create(version=version, input=input_data)
    print(f"   -> Replicate задача создана с ID: {prediction.id}")
    start_time = time.time()
    timeout = 600
    while True:
        try:
            print(f"   -> Запрос актуального статуса для '{description}' (ID: {prediction.id})...")
            prediction = replicate_client.predictions.get(prediction.id)
            print(f"   -> Статус получен: {prediction.status}")
        except Exception as e:
            print(f"   -> !!! ВНИМАНИЕ: Ошибка при получении статуса для '{description}': {e}. Повторная попытка...")
            time.sleep(5)
            continue
        if prediction.status in ["succeeded", "failed", "canceled"]: break
        if time.time() - start_time > timeout: raise Exception(f"Модель '{description}' не завершилась за {timeout} секунд (таймаут).")
        time.sleep(3)
    if prediction.status != 'succeeded': raise Exception(f"Модель '{description}' не удалась со статусом {prediction.status}. Ошибка: {prediction.error}")
    print(f"<- Модель '{description}' успешно завершена.")
    return prediction.output

# ЗАМЕНИТЕ ВАШУ СТАРУЮ ФУНКЦИЮ НА ЭТУ

# ЗАМЕНИТЕ ВАШУ СТАРУЮ ФУНКЦИЮ ЦЕЛИКОМ НА ЭТУ ВЕРСИЮ

def composite_images(original_url, upscaled_url, mask_url):
    print("-> Начало композитинга изображений (метод OpenCV)...")
    try:
        # --- ШАГ 1: ЗАГРУЗКА ИЗОБРАЖЕНИЙ ---
        # Вспомогательная функция для загрузки картинки по URL
        def url_to_image(url, flags=cv2.IMREAD_UNCHANGED):
            resp = requests.get(url, stream=True).raw
            image_array = np.asarray(bytearray(resp.read()), dtype="uint8")
            return cv2.imdecode(image_array, flags)
            
        original_img_bgr = url_to_image(original_url, cv2.IMREAD_COLOR)
        upscaled_img_bgr = url_to_image(upscaled_url, cv2.IMREAD_COLOR)
        mask_img_gray = url_to_image(mask_url, cv2.IMREAD_GRAYSCALE)

        # Конвертируем изображения в формат с альфа-каналом (прозрачностью) для корректной работы
        original_img = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2BGRA)
        upscaled_img = cv2.cvtColor(upscaled_img_bgr, cv2.COLOR_BGR2BGRA)
        
        # --- ШАГ 2: ПОДГОТОВКА РАЗМЕРОВ ---
        # Вспомогательная функция для подгонки размера с сохранением пропорций
        def resize_with_padding(img, target_shape):
            target_h, target_w = target_shape[:2]
            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if len(img.shape) > 2:
                channels = img.shape[2]
                canvas = np.full((target_h, target_w, channels), (0, 0, 0, 0), dtype=np.uint8) # Прозрачный фон
            else:
                canvas = np.zeros((target_h, target_w), dtype=np.uint8)
            
            x_center = (target_w - new_w) // 2
            y_center = (target_h - new_h) // 2
            canvas[y_center:y_center + new_h, x_center:x_center + new_w] = resized
            return canvas

        # Эталоном размера является изображение после апскейла
        target_shape = upscaled_img.shape

        print("   -> Подгонка слоев к единому размеру...")
        original_resized = resize_with_padding(original_img, target_shape)
        mask_resized_padded = resize_with_padding(mask_img_gray, target_shape)

        # --- ШАГ 3: ОБРАБОТКА МАСКИ (БЕЗ РАСШИРЕНИЯ) ---
        h, w = target_shape[:2]
        # --- ИЗМЕНЕНИЕ: Возвращаем расширение маски с новыми параметрами ---
        # Шаг А: Расширяем маску (Dilation)
        # Задаем размер расширения. 1.5% от ширины - хорошее начало.
        # Можете менять это значение (например, на 0.02 для большего расширения).
        expand_size = int(w * 0.05)
        # Ядро для операции должно иметь нечетный размер
        expand_size = expand_size if expand_size % 2 != 0 else expand_size + 1 
        
        # Создаем "ядро" для операции расширения
        kernel = np.ones((expand_size, expand_size), np.uint8)
        
        # Применяем расширение: белые области на маске "растут"
        expanded_mask = cv2.dilate(mask_resized_padded, kernel, iterations=1)

        # Шаг Б: Смягчаем края (Blur)
        # Размер размытия делаем зависимым от размера расширения для предсказуемости
        blur_size = int(expand_size * 0.5) # Например, половина от размера расширения
        blur_size = blur_size if blur_size % 2 != 0 else blur_size + 1

        # Применяем размытие Гаусса к НОВОЙ, РАСШИРЕННОЙ маске
        soft_mask = cv2.GaussianBlur(expanded_mask, (blur_size, blur_size), 0)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # --- ШАГ 4: СОЗДАНИЕ АЛЬФА-КАНАЛА ДЛЯ СМЕШИВАНИЯ ---
        # Нормализуем значения маски от 0 (черный) до 1.0 (белый)
        alpha_mask_float = (soft_mask / 255.0).astype(np.float32)
        # Добавляем новую ось, чтобы NumPy мог применить эту одноканальную маску ко всем 4 каналам RGBA
        alpha_for_blending = alpha_mask_float[..., np.newaxis]
        
        # --- ШАГ 5: ФИНАЛЬНЫЙ КОМПОЗИТИНГ ---
        # Формула альфа-смешивания: Итог = (ВерхнийСлой * Маска) + (НижнийСлой * (1 - Маска))
        composite = (alpha_for_blending * upscaled_img.astype(np.float32)) + ((1 - alpha_for_blending) * original_resized.astype(np.float32))
        final_image = composite.astype(np.uint8)
        
        print(f"<- Композитинг (OpenCV) успешно завершен. Финальное разрешение: {final_image.shape[1]}x{final_image.shape[0]}")
        
        _, image_data_encoded = cv2.imencode('.png', final_image)
        image_data = io.BytesIO(image_data_encoded)
        image_data.seek(0)
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

def process_job(job_data):
    prediction_id = job_data.get('prediction_id')
    app_base_url = job_data.get('app_base_url')

    print(f"--- Начало обработки задачи {prediction_id} от {app_base_url} ---")

    final_payload = {"prediction_id": prediction_id}
    if not app_base_url:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: app_base_url не передан в задаче {prediction_id}")
        return

    webhook_url = f"{app_base_url.rstrip('/')}/worker-webhook"
    headers = {"Authorization": f"Bearer {WORKER_SECRET_KEY}", "Content-Type": "application/json"}
    try:
        flux_output = run_replicate_model(FLUX_MODEL_VERSION, {"input_image": job_data['original_s3_url'], "prompt": job_data['generation_prompt']}, "FLUX Edit")
        generated_image_url = flux_output[0] if isinstance(flux_output, list) else flux_output

        # --- ИЗМЕНЕНИЕ: Реализована ваша логика выбора изображения для маски ---
        intent = job_data.get('intent')
        # По умолчанию маску создаем по ОРИГИНАЛЬНОМУ изображению
        image_for_masking = job_data['original_s3_url']
        # Но если мы ДОБАВЛЯЕМ или ЗАМЕНЯЕМ объект, то маску нужно искать на НОВОМ изображении
        if intent in ['ADD', 'REPLACE']:
            image_for_masking = generated_image_url
            print(f"   -> Создание маски по НОВОМУ изображению (intent: {intent})")
        else: # Для 'REMOVE' или если intent не определен
            print(f"   -> Создание маски по ОРИГИНАЛЬНОМУ изображению (intent: {intent})")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        sam_input = {"image": image_for_masking, "text_prompt": job_data['mask_prompt']}
        mask_output = run_replicate_model(SAM_MODEL_VERSION, sam_input, "Lang-SAM Masking")
        mask_url = mask_output[0] if isinstance(mask_output, list) else mask_output
        
        original_width = job_data.get('original_width', 1024)
        if original_width <= 2048: scale_factor, creativity, resemblance, hdr, num_inference_steps = 2.0, 0.30, 1.60, 3, 50
        elif original_width <= 4096: scale_factor, creativity, resemblance, hdr, num_inference_steps = 4.0, 0.30, 1.60, 3, 60
        else: scale_factor, creativity, resemblance, hdr, num_inference_steps = 4.0, 0.30, 1.60, 3, 60
        
        upscaler_input = { "image": generated_image_url, "scale_factor": scale_factor, "creativity": creativity, "resemblance": resemblance, "num_inference_steps": num_inference_steps, "dynamic": hdr }
        upscaled_output = run_replicate_model(UPSCALER_MODEL_VERSION, upscaler_input, "Upscaler")
        upscaled_image_url = upscaled_output[0] if isinstance(upscaled_output, list) else upscaled_output

        final_image_data = composite_images(job_data['original_s3_url'], upscaled_image_url, mask_url)
        final_s3_url = upload_to_s3(final_image_data, job_data['user_id'], prediction_id)
        
        final_payload['status'] = 'completed'
        final_payload['final_url'] = final_s3_url
        print(f"--- Задача {prediction_id} УСПЕШНО ЗАВЕРШЕНА! Отправка результата в app...")

    except Exception as e:
        print(f"!!! ОШИБКА при обработке задачи {prediction_id}:")
        traceback.print_exc()
        final_payload['status'] = 'failed'

    finally:
        try:
            requests.post(webhook_url, json=final_payload, headers=headers, timeout=20)
            print(f"--- Результат для задачи {prediction_id} отправлен в {webhook_url} ---")
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось отправить вебхук в app для задачи {prediction_id}: {e}")

def main_loop():
    print(">>> Воркер PiflyEdit запущен и ожидает задач...")
    redis_client = redis.from_url(REDIS_URL)
    while True:
        try:
            _, job_json = redis_client.brpop('pifly_edit_jobs', 0)
            job_data = json.loads(job_json)
            print(f"--- WORKER: Получена новая задача: {job_data.get('prediction_id')} ---")
            process_job(job_data)
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА в основном цикле воркера: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
