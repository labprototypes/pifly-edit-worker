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

def composite_images(original_url, upscaled_url, mask_url):
    print("-> Начало композитинга изображений...")
    try:
        original_img = Image.open(requests.get(original_url, stream=True).raw).convert("RGBA")
        upscaled_img = Image.open(requests.get(upscaled_url, stream=True).raw).convert("RGBA")
        mask_img = Image.open(requests.get(mask_url, stream=True).raw).convert("L")

        # --- НОВЫЙ БЛОК ПРОДВИНУТОЙ ОБРАБОТКИ МАСКИ ---
        # Шаг 1: Рассчитываем, на сколько пикселей расширить маску (примерно 15% от ширины)
        expand_size = int(mask_img.width * 0.15) # 15% в каждую сторону = 30% общая ширина
        # Убедимся, что размер нечетный для фильтра
        expand_size = expand_size if expand_size % 2 != 0 else expand_size + 1
        print(f"   -> Расширяем маску на ~{expand_size} пикселей...")

        # Расширяем маску, "наращивая" белые области
        expanded_mask = mask_img.filter(ImageFilter.MaxFilter(size=expand_size))

        # Шаг 2: Рассчитываем радиус размытия для растушевки (примерно 20% от размера расширения)
        blur_radius = int(expand_size * 0.2)
        print(f"   -> Растушевываем края маски с радиусом: {blur_radius}")

        # Размываем расширенную маску для создания мягких краев
        soft_mask = expanded_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # --- КОНЕЦ НОВОГО БЛОКА ---

        # Остальная логика остается прежней, но использует новую soft_mask
        high_res_mask = soft_mask.resize(upscaled_img.size, Image.LANCZOS)
        high_res_original = original_img.resize(upscaled_img.size, Image.LANCZOS)
        high_res_final = Image.composite(upscaled_img, high_res_original, high_res_mask)
        final_image = high_res_final.resize(original_img.size, Image.LANCZOS)

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
    prediction_id = job_data['prediction_id']
    print(f"--- Начало обработки задачи {prediction_id} ---")
    try:
        # Шаг 1: Генерация FLUX
        flux_output = run_replicate_model(FLUX_MODEL_VERSION, {"input_image": job_data['original_s3_url'], "prompt": job_data['generation_prompt']}, "FLUX Edit")
        generated_image_url = flux_output[0] if isinstance(flux_output, list) else flux_output

        # --- НОВАЯ ЛОГИКА ВЫБОРА ИСТОЧНИКА ДЛЯ МАСКИ ---
        intent = job_data.get('intent')
        if intent == 'ADD':
            # Если ДОБАВЛЯЕМ, ищем объект на НОВОЙ картинке
            image_for_masking = generated_image_url
            print(f"   -> Режим 'ADD'. Маска будет создаваться по новому изображению.")
        else: # Для REMOVE и REPLACE
            # Если УДАЛЯЕМ или ЗАМЕНЯЕМ, ищем объект на ОРИГИНАЛЬНОЙ картинке
            image_for_masking = job_data['original_s3_url']
            print(f"   -> Режим '{intent}'. Маска будет создаваться по оригинальному изображению.")

        # Шаг 2: Создание маски
        sam_input = {"image": image_for_masking, "text_prompt": job_data['mask_prompt']}
        mask_output = run_replicate_model(SAM_MODEL_VERSION, sam_input, "Lang-SAM Masking")
        mask_url = mask_output[0] if isinstance(mask_output, list) else mask_output

        # ... (остальные шаги - апскейл, композитинг, сохранение - без изменений) ...

        upscaled_output = run_replicate_model(UPSCALER_MODEL_VERSION, {"image": generated_image_url}, "Upscaler")
        upscaled_image_url = upscaled_output[0] if isinstance(upscaled_output, list) else upscaled_output
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
