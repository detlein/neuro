import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from data_collector import log_interaction
from data_preprocessing import preprocess_dialogues, preprocess_text
from model_architecture import load_model_and_tokenizer, prepare_data
import post_training
from post_training import post_train_model

CLEANED_DATA_PATH = 'cleaned_dialogues.json'
MODEL_PATH = 'trained_model.pth'
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 5e-5
NUM_BEAMS = 5  # Число лучей для улучшенного поиска


def collate_fn(batch):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # или другой специальный токен
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    input_ids_batch, label_ids_batch = zip(*batch)
    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=float(tokenizer.pad_token_id))
    label_ids_batch = pad_sequence(label_ids_batch, batch_first=True, padding_value=-100.0)

    max_len = max(input_ids_batch.size(1), label_ids_batch.size(1))
    if input_ids_batch.size(1) < max_len:
        padding = torch.full((input_ids_batch.size(0), max_len - input_ids_batch.size(1)), tokenizer.pad_token_id).to(
            input_ids_batch.device)
        input_ids_batch = torch.cat((input_ids_batch, padding), dim=1)
    if label_ids_batch.size(1) < max_len:
        padding = torch.full((label_ids_batch.size(0), max_len - label_ids_batch.size(1)), -100).to(
            label_ids_batch.device)
        label_ids_batch = torch.cat((label_ids_batch, padding), dim=1)

    return input_ids_batch, label_ids_batch


def generate_response(model, tokenizer, prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    attention_mask = torch.ones(input_ids.shape, device=model.device)
    if tokenizer.pad_token_id is not None:
        input_ids[input_ids == tokenizer.pad_token_id] = 0
        attention_mask[input_ids != 0] = 1

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            num_beams=NUM_BEAMS,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Запуск диалога:")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Файл trained_model.pth не найден, пожалуйста, сначала обучите модель.")
        return

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    try:
        while True:
            user_input = input("Вы: ")
            if user_input.lower() == "exit":
                print("Диалог завершён. Все данные сохранены.")
                preprocess_dialogues()
                break

            bot_response = generate_response(model, tokenizer, user_input)
            print(f"Бот: {bot_response}")
            log_interaction(user_input, bot_response)

            preprocess_dialogues()  # Обработка новых данных после каждой сессии

            dialogues = post_training.load_new_data(CLEANED_DATA_PATH)
            dataset = post_training.NewDialogueDataset(tokenizer, dialogues)
            train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            post_train_model(model, tokenizer, train_loader, optimizer, scheduler, device)

            torch.save(model.state_dict(), MODEL_PATH)
            print("Model post-trained and saved successfully.")

    except KeyboardInterrupt:
        print("\nДиалог завершён. Все данные сохранены.")
        preprocess_dialogues()


if __name__ == "__main__":
    main()
