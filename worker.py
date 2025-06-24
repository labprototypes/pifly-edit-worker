# worker.py - ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ

import os, time, json, requests, io, traceback, uuid, redis, boto3
import cv2
import numpy as np
from PIL import Image, ImageFilter
import replicate

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

# Модели Replicate
FLUX_MODEL_VERSION = "black-forest-labs/flux-kontext-max:0b9c317b23e79a9a0d8b9602ff4d04030d433055927fb7c4b91c44234a6818c4"
SAM_MODEL_VERSION = "tmappdev/lang-segment-anything:891411c38a6ed2d44c004b7b9e44217df7a5b07848f29ddefd2e28bc7cbf93bc"
UPSCALER_MODEL_VERSION = "philz1337x/clarity-upscaler:dfad41707589d68ecdccd1dfa600d55a208f9310748e44bfe35b4a6291453d5e"

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN, timeout=180.0)

# Создание независимого подключения к БД
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Определение моделей БД прямо в файле для автономности
class User(Base):
    __tablename__ = 'user'
    id = Column(String(128), primary_key=True)
    token_balance = Column(Integer, nullable=False)

class Prediction(Base):
    __tablename__ = 'prediction'
    id = Column(String(36), primary_key=True)
    user_id = Column(String(128), nullable=False)
    status = Column(String(50), nullable=False)
    output_url = Column(String(2048), nullable=True)
    token_cost = Column(Integer, nullable=False)

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

def composite_images(original_url, upscaled_url, mask_url):
    print("-> Начало композитинга изображений (метод OpenCV)...")
    try:
        MAX_RESOLUTION = 4096

        def url_to_image(url, flags=cv2.IMREAD_UNCHANGED):
            resp = requests.get(url, stream=True).raw
            image_array = np.asarray(bytearray(resp.read()), dtype="uint8")
            return cv2.imdecode(image_array, flags)
            
        def resize_with_padding(img, target_shape):
            target_h, target_w = target_shape[:2]
            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if len(img.shape) > 2:
                channels = img.shape[2]
                canvas = np.full((target_h, target_w, channels), (0, 0, 0, 255), dtype=np.uint8)
            else:
                canvas = np.zeros((target_h, target_w), dtype=np.uint8)
            
            x_center = (target_w - new_w) // 2
            y_center = (target_h - new_h) // 2
            canvas[y_center:y_center + new_h, x_center:x_center + new_w] = resized
            return canvas

        original_img_bgr = url_to_image(original_url, cv2.IMREAD_COLOR)
        upscaled_img_bgr = url_to_image(upscaled_url, cv2.IMREAD_COLOR)
        mask_img_gray = url_to_image(mask_url, cv2.IMREAD_GRAYSCALE)
        original_img = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2BGRA)
        upscaled_img = cv2.cvtColor(upscaled_img_bgr, cv2.COLOR_BGR2BGRA)
        target_shape = upscaled_img.shape

        if max(target_shape[:2]) > MAX_RESOLUTION:
            scale = MAX_RESOLUTION / max(target_shape[:2])
            new_w, new_h = int(target_shape[1] * scale), int(target_shape[0] * scale)
            print(f"   -> Изображение слишком большое. Ограничиваем до {new_w}x{new_h}px.")
            upscaled_img = cv2.resize(upscaled_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            target_shape = upscaled_img.shape

        print("   -> Подгонка слоев к единому размеру без искажений...")
        original_resized = resize_with_padding(original_img, target_shape)
        mask_resized_padded = resize_with_padding(mask_img_gray, target_shape)

        h, w = target_shape[:2]
        
        # --- ИСПРАВЛЕННАЯ ЛОГИКА БЕЗ РАСШИРЕНИЯ МАСКИ ---
        # Вычисляем размер размытия напрямую от ширины изображения (например, 1% от ширины)
        blur_size = int(w * 0.05)
        # Убедимся, что размер ядра нечетный, как того требует GaussianBlur
        blur_size = blur_size if blur_size % 2 != 0 else blur_size + 1

        # Применяем размытие напрямую к оригинальной (подогнанной по размеру) маске
        soft_mask = cv2.GaussianBlur(mask_resized_padded, (blur_size, blur_size), 0)
        # --- КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ ---

        soft_mask_float = soft_mask.astype(np.float32) / 255.0
        soft_mask_alpha = cv2.cvtColor(soft_mask_float, cv2.COLOR_GRAY2BGRA)

        composite = (soft_mask_alpha * upscaled_img.astype(np.float32)) + ((1 - soft_mask_alpha) * original_resized.astype(np.float32))
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
    prediction_id = job_data['prediction_id']
    print(f"--- Начало обработки задачи {prediction_id} ---")
    db_session = get_db_session()
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

        prediction = db_session.get(Prediction, prediction_id)
        if prediction:
            prediction.status = 'completed'
            prediction.output_url = final_s3_url
            db_session.commit()
            print(f"--- ПОЛНАЯ ЗАДАЧА {prediction_id} УСПЕШНО ЗАВЕРШЕНА! ---")
    except Exception as e:
        print(f"!!! ОШИБКА при обработке задачи {prediction_id}:")
        traceback.print_exc()
        db_session.rollback()
        prediction = db_session.get(Prediction, prediction_id)
        if prediction:
            prediction.status = 'failed'
            user = db_session.get(User, prediction.id)
            if user:
                user.token_balance += prediction.token_cost
                print(f"Возвращено {prediction.token_cost} токенов пользователю {user.id}")
            db_session.commit()
    finally:
        db_session.close()

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
