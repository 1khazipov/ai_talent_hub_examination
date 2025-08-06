import logging
import json
import os
import requests
import numpy as np
import faiss

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes,
)

# Импортируем наш модуль рекомендаций
from recommender import get_recommendation

from dotenv import load_dotenv

load_dotenv()

# ---- 1. НАСТРОЙКИ И КОНСТАНТЫ ----

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Токен вашего бота
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Настройки для Ollama
OLLAMA_API_URL = "http://localhost:11434/api/"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3:8b"

# Пути к файлам базы знаний
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "text_chunks.json")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

# Состояния для диалогов
STATE_ASK_BACKGROUND = 1
STATE_ASK_QUESTION = 2

# ---- 2. ЗАГРУЗКА БАЗЫ ЗНАНИЙ ----

try:
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        text_chunks = json.load(f)
    logger.info(f"База знаний успешно загружена: {faiss_index.ntotal} векторов.")
except Exception as e:
    logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить базу знаний: {e}")
    logger.error("Убедитесь, что вы запустили parser.py, а затем create_knowledge_base.py перед стартом бота.")
    faiss_index = None
    text_chunks = None

# ---- 3. ФУНКЦИИ ДЛЯ РАБОТЫ С OLLAMA И RAG ----

def get_embedding(text: str) -> np.ndarray | None:
    """Получает эмбеддинг для текста через API Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return np.array(response.json().get("embedding"), dtype='float32')
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении эмбеддинга от Ollama: {e}")
        return None
    except (KeyError, TypeError):
        logger.error("Не удалось извлечь эмбеддинг из ответа Ollama.")
        return None

def find_relevant_chunks(question: str, top_k: int = 5) -> str:
    """Находит релевантные чанки в базе знаний и возвращает их как единый контекст."""
    if not faiss_index or not text_chunks:
        return "База знаний недоступна."

    question_embedding = get_embedding(question)
    if question_embedding is None:
        return "Не удалось обработать ваш вопрос для поиска по базе знаний."

    # FAISS требует 2D-массив для поиска
    question_embedding_np = np.array([question_embedding])
    
    distances, indices = faiss_index.search(question_embedding_np, top_k)
    
    context = ""
    for i in indices[0]:
        if i != -1:
            chunk = text_chunks[i]
            context += f"Фрагмент из описания программы '{chunk['program_name']}':\n---\n{chunk['text']}\n---\n\n"
    
    return context.strip() if context else "В базе знаний не найдено релевантной информации."

def get_llm_response(question: str, context: str) -> str:
    """Отправляет запрос к языковой модели с вопросом и контекстом."""
    prompt = f"""
Ты — дружелюбный и экспертный ИИ-ассистент для абитуриентов ИТМО.
Твоя задача — максимально точно и полно ответить на вопрос пользователя, используя ТОЛЬКО предоставленный ниже контекст.
Не придумывай информацию. Если в контексте нет прямого ответа, скажи, что не можешь найти точную информацию по этому вопросу в доступных материалах.
Отвечай структурированно и по делу.

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ТВОЙ ОТВЕТ:
"""
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "think": False},
            timeout=120  # Увеличим таймаут для генерации
        )
        response.raise_for_status()
        return response.json().get('response', "Модель не дала ответа.").strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе к LLM Ollama: {e}")
        return "Извините, произошла ошибка при обращении к языковой модели. Попробуйте позже."


# ---- 4. ОБРАБОТЧИКИ КОМАНД И ДИАЛОГОВ ----

def get_main_menu_keyboard():
    keyboard = [
        ["Сравнить программы"],
        ["Задать вопрос по программам"],
        ["Помоги выбрать (Рекомендация)"],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\n\n"
        "Я бот-помощник для абитуриентов магистратур ИТМО по AI. "
        "Готов помочь сравнить программы, ответить на вопросы или дать рекомендацию.",
        reply_markup=get_main_menu_keyboard(),
    )

async def compare_programs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = "Сравни магистерские программы 'Искусственный интеллект' и 'Управление AI-продуктами'. Опиши их ключевые цели, для кого они подходят, и кем становятся выпускники. Представь ответ в виде сравнения по пунктам."
    await update.message.reply_text("Готовлю сравнение программ на основе данных с сайтов... Это может занять минуту.")
    
    relevant_context = find_relevant_chunks(question, top_k=10) # Берем больше чанков для полноты
    if "База знаний недоступна" in relevant_context or "Не удалось обработать" in relevant_context:
        await update.message.reply_text(relevant_context)
        return
        
    answer = get_llm_response(question, relevant_context)
    await update.message.reply_text(answer)

# --- Блок рекомендаций ---
async def recommendation_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Конечно! Чтобы я дал рекомендацию, опишите ваш текущий опыт или образование в 1-2 предложениях.\n\n"
        "Например: 'Я frontend-разработчик, хочу в ML' или 'Я менеджер в IT-компании'.\n\nДля отмены введите /cancel",
    )
    return STATE_ASK_BACKGROUND

async def process_background(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    recommendation = get_recommendation(update.message.text)
    await update.message.reply_text(recommendation, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
    return ConversationHandler.END

# --- Блок вопросов к RAG ---
async def question_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Задайте свой вопрос о поступлении, учебном процессе, дисциплинах или карьере. Я поищу ответ в материалах с сайтов обеих программ.\n\nДля отмены введите /cancel"
    )
    return STATE_ASK_QUESTION

async def process_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    question = update.message.text
    await update.message.reply_text("Ищу информацию и генерирую ответ... Пожалуйста, подождите.")
    
    relevant_context = find_relevant_chunks(question)
    if "База знаний недоступна" in relevant_context or "Не удалось обработать" in relevant_context:
        await update.message.reply_text(relevant_context, reply_markup=get_main_menu_keyboard())
        return ConversationHandler.END

    answer = get_llm_response(question, relevant_context)
    await update.message.reply_text(answer, reply_markup=get_main_menu_keyboard())
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Действие отменено.", reply_markup=get_main_menu_keyboard())
    return ConversationHandler.END


# ---- 5. ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА БОТА ----

def main() -> None:
    """Запуск бота."""
    if not faiss_index:
        logger.critical("Бот не может быть запущен без базы знаний. Завершение работы.")
        return
        
    if TELEGRAM_BOT_TOKEN == "ВАШ_ТЕЛЕГРАМ_ТОКЕН_ЗДЕСЬ":
        logger.critical("Необходимо указать токен Telegram-бота в переменной TELEGRAM_BOT_TOKEN.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- Диалог для рекомендаций ---
    rec_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex("^Помоги выбрать \(Рекомендация\)$"), recommendation_start)],
        states={
            STATE_ASK_BACKGROUND: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_background)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # --- Диалог для вопросов ---
    question_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex("^Задать вопрос по программам$"), question_start)],
        states={
            STATE_ASK_QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_question)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    # Добавляем обработчики в приложение
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.Regex("^Сравнить программы$"), compare_programs_command))
    application.add_handler(rec_handler)
    application.add_handler(question_handler)

    logger.info("Бот запущен и готов к работе...")
    application.run_polling()

if __name__ == '__main__':
    main()
