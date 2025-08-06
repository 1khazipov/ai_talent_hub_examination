import time
import json
import os
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

PROGRAM_URLS = {
    "ai": { "name": "Искусственный интеллект", "url": "https://abit.itmo.ru/program/master/ai" },
    "ai_product": { "name": "Управление AI-продуктами", "url": "https://abit.itmo.ru/program/master/ai_product" },
}
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "text_chunks.json")

def _simple_sentence_split(text):
    """Надежно разбивает текст на предложения без внешних библиотек."""
    text = re.sub(r'\. ', '.<SPLIT>', text)
    text = re.sub(r'\? ', '?<SPLIT>', text)
    text = re.sub(r'\! ', '!<SPLIT>', text)
    sentences = text.split('<SPLIT>')
    return [s.strip() for s in sentences if s and len(s.split()) > 2]

# --- НОВАЯ ФУНКЦИЯ ДЛЯ УМНОГО СОЗДАНИЯ ЧАНКОВ ---
def create_sized_chunks(text, min_length=128, max_length=512):
    """
    Создает чанки, длина которых находится в заданном диапазоне.
    """
    sentences = _simple_sentence_split(text)
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        
        # Если добавление нового предложения превысит лимит, сохраняем текущий чанк
        if current_length + sentence_len + (1 if current_chunk_sentences else 0) > max_length:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_length = sentence_len
        else:
            current_chunk_sentences.append(sentence)
            current_length += sentence_len + (1 if len(current_chunk_sentences) > 1 else 0)
    
    # Добавляем последний оставшийся чанк
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    # --- Финальный проход для объединения слишком коротких чанков ---
    final_chunks = []
    buffer = ""
    for chunk in chunks:
        if buffer:
            potential_buffer = buffer + " " + chunk
        else:
            potential_buffer = chunk

        if len(potential_buffer) <= max_length:
            buffer = potential_buffer
        else:
            final_chunks.append(buffer)
            buffer = chunk
    
    if buffer:
        final_chunks.append(buffer)
        
    # Возвращаем только те чанки, которые соответствуют минимальной длине
    return [c for c in final_chunks if len(c) >= min_length]


def _clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless"); options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage"); options.add_argument("--window-size=1920,1080")
    service = ChromeService(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def parse_page(soup, program_name):
    all_program_chunks = []
    main_container = soup.find('main')
    if not main_container: main_container = soup

    # Ключевые факты - они короткие и идут как есть
    params_list = main_container.find('ul', {'data-testid': 'program-params-list'})
    if params_list:
        for item in params_list.find_all('li'):
            key = item.find('p', class_=lambda c: c and 'title' in c)
            value = item.find('p', class_=lambda c: c and 'value' in c)
            if key and value: all_program_chunks.append(f"{_clean_text(key.text)}: {_clean_text(value.text)}.")

    # Большие текстовые секции, которые нужно правильно поделить
    sections = main_container.find_all('h2', class_=lambda c: c and 'title' in c)
    for header in sections:
        header_text = _clean_text(header.text)
        container = header.find_parent().find_parent()
        if container:
            content_text = _clean_text(container.get_text(separator=' ')).replace(header_text, "", 1).strip()
            if len(content_text) > 50: # Обрабатываем только большие секции
                # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
                sized_chunks = create_sized_chunks(content_text)
                for chunk in sized_chunks:
                    all_program_chunks.append(f"Из раздела '{header_text}': {chunk}")
                    
    return all_program_chunks

def main():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    all_chunks = []; driver = setup_driver()
    try:
        for key, info in PROGRAM_URLS.items():
            print(f"\n--- Парсинг программы: {info['name']} ---")
            driver.get(info['url'])
            time.sleep(5)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            program_chunks = parse_page(soup, info['name'])
            print(f"  - Собрано и разделено на {len(program_chunks)} фрагментов (длина 128-512 симв.).")
            for i, chunk_text in enumerate(program_chunks):
                all_chunks.append({"source": info['url'], "program_name": info['name'], "text": chunk_text, "chunk_id": f"{key}_{i}"})
    finally:
        driver.quit()
    if not all_chunks: print("\n!!! ОШИБКА: Не удалось собрать данные."); return
    try:
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)
        print(f"\n>>> УСПЕХ! Данные ({len(all_chunks)} чанков) собраны и сохранены в: {CHUNKS_FILE}")
    except IOError as e:
        print(f"\n!!! Ошибка при записи в файл: {e}")

if __name__ == "__main__":
    main()
