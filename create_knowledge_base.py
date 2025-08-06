import json
import requests
import numpy as np
import faiss
import os

OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "text_chunks.json")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

def get_embedding(text: str) -> np.ndarray:
    """Получает эмбеддинг для текста через API Ollama."""
    try:
        payload = {
            "model": EMBEDDING_MODEL,
            "prompt": text
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        embedding = response.json().get("embedding")
        return np.array(embedding, dtype='float32')
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении эмбеддинга: {e}")
        return None
    except (KeyError, TypeError):
        print("Не удалось извлечь эмбеддинг из ответа Ollama.")
        return None

def main():
    print("Запуск создания базы знаний...")

    # 1. Загрузка текстовых чанков
    try:
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Файл {CHUNKS_FILE} не найден. Сначала запустите parser.py.")
        return
    
    if not chunks:
        print("Нет данных для создания базы знаний.")
        return

    print(f"Загружено {len(chunks)} текстовых чанков.")

    # 2. Получение эмбеддингов для каждого чанка
    embeddings = []
    valid_chunks = [] # Сохраняем только те чанки, для которых удалось создать эмбеддинг
    for i, chunk_data in enumerate(chunks):
        print(f"Обработка чанка {i+1}/{len(chunks)}...")
        embedding = get_embedding(chunk_data['text'])
        if embedding is not None:
            embeddings.append(embedding)
            valid_chunks.append(chunk_data)
        else:
            print(f"Пропуск чанка из-за ошибки с эмбеддингом: {chunk_data['chunk_id']}")
            
    if not embeddings:
        print("Не удалось создать ни одного эмбеддинга. Прерывание.")
        return
        
    # Преобразование списка эмбеддингов в numpy-массив
    embeddings_np = np.array(embeddings)
    
    # 3. Создание и обучение индекса FAISS
    dimension = embeddings_np.shape[1]  # Размерность векторов
    index = faiss.IndexFlatL2(dimension)   # Используем L2 расстояние для поиска
    index.add(embeddings_np)
    
    print(f"\nСоздан FAISS индекс с {index.ntotal} векторами размерности {dimension}.")
    
    # 4. Сохранение индекса и валидных чанков
    faiss.write_index(index, FAISS_INDEX_FILE)
    # Перезаписываем файл чанков, чтобы он соответствовал индексам в FAISS
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(valid_chunks, f, ensure_ascii=False, indent=4)
        
    print(f"Индекс сохранен в: {FAISS_INDEX_FILE}")
    print(f"Соответствующие чанки обновлены в: {CHUNKS_FILE}")
    print("\nБаза знаний успешно создана!")


if __name__ == "__main__":
    main()
