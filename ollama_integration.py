import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:8b"

def get_ollama_response(question: str, context: str) -> str:
    """
    Отправляет запрос к модели Ollama с вопросом и контекстом, возвращает ответ.
    """
    prompt = f"""
    Ты — ИИ-ассистент, который помогает абитуриентам ИТМО.
    Твоя задача — отвечать на вопросы, основываясь ИСКЛЮЧЕНиеЛЬНО на предоставленном контексте.
    Не придумывай информацию. Если в контексте нет ответа, так и скажи: "В предоставленной информации нет ответа на этот вопрос".
    Отвечай на русском языке.

    Контекст:
    ---
    {context}
    ---

    Вопрос абитуриента: "{question}"

    Ответ:
    """

    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False  # Получаем ответ целиком, а не потоком
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        # Ответ от Ollama приходит в формате JSON, где каждая строка - это отдельный JSON объект.
        # Так как stream=False, ответ должен быть одним JSON объектом.
        response_data = response.json()
        
        return response_data.get('response', "Не удалось получить осмысленный ответ от модели.").strip()

    except requests.exceptions.ConnectionError:
        return "Ошибка: Не удалось подключиться к Ollama. Убедитесь, что Ollama запущен и доступен по адресу http://localhost:11434."
    except requests.exceptions.RequestException as e:
        return f"Произошла ошибка при запросе к Ollama: {e}"

# Пример использования (для локального теста)
if __name__ == '__main__':
    test_context = "Программа 'Искусственный интеллект' готовит ML-инженеров. Стоимость обучения 599 000 рублей в год."
    test_question = "Сколько стоит обучение на программе ИИ?"
    
    print("Отправка тестового запроса в Ollama...")
    answer = get_ollama_response(test_question, test_context)
    print("\nОтвет модели:")
    print(answer)
