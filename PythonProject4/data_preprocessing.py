import json
import re

CLEANED_DATA_PATH = 'cleaned_dialogues.json'

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dialogues(log_file='dialogues_log.json', cleaned_file=CLEANED_DATA_PATH):
    with open(log_file, 'r') as file:
        dialogues = json.load(file)

    cleaned_dialogues = []

    for dialogue in dialogues:
        cleaned_entry = {
            'timestamp': dialogue['timestamp'],
            'user_input': preprocess_text(dialogue['user_input']),
            'bot_response': preprocess_text(dialogue['bot_response'])
        }
        cleaned_dialogues.append(cleaned_entry)

    with open(cleaned_file, 'w') as file:
        json.dump(cleaned_dialogues, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    preprocess_dialogues()
