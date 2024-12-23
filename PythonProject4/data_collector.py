import os
import json
from datetime import datetime

LOG_FILE_PATH = 'dialogues_log.json'

def log_interaction(user_input, bot_response):
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w', encoding='utf-8') as file:
            json.dump([], file, ensure_ascii=False, indent=4)

    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as file:
            logs = json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logs = []
        print(f"Warning: Could not read the log file. Starting with an empty log. Error: {e}")

    dialogue_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_input': user_input,
        'bot_response': bot_response
    }

    logs.append(dialogue_entry)

    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as file:
        json.dump(logs, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    user_input = "Привет!"
    bot_response = "Привет! Как я могу помочь тебе сегодня?"
    log_interaction(user_input, bot_response)
