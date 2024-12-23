import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import os

CLEANED_DATA_PATH = 'cleaned_dialogues.json'
MODEL_PATH = 'trained_model.pth'
NEW_DATA_PATH = 'new_dialogues.json'
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 5e-5

class NewDialogueDataset(Dataset):
    def __init__(self, tokenizer, dialogues):
        self.inputs = []
        self.labels = []

        for dialogue in dialogues:
            input_ids = tokenizer.encode(dialogue['user_input'], add_special_tokens=True)
            label_ids = tokenizer.encode(dialogue['bot_response'], add_special_tokens=True)
            self.inputs.append(torch.tensor(input_ids).long())  # Убедимся, что это torch.LongTensor
            self.labels.append(torch.tensor(label_ids).long())  # Убедимся, что это torch.LongTensor

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def load_new_data(file_path=NEW_DATA_PATH):
    with open(file_path, 'r') as file:
        return json.load(file)


def collate_fn(batch):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Убедимся, что pad_token_id инициализирован
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # или другой специальный токен
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    input_ids_batch, label_ids_batch = zip(*batch)
    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True, padding_value=float(tokenizer.pad_token_id))
    label_ids_batch = pad_sequence(label_ids_batch, batch_first=True, padding_value=-100.0)

    # Убедимся, что входные и целевые тензоры одинакового размера
    max_len = max(input_ids_batch.size(1), label_ids_batch.size(1))
    if input_ids_batch.size(1) < max_len:
        padding = torch.full((input_ids_batch.size(0), max_len - input_ids_batch.size(1)), tokenizer.pad_token_id).to(input_ids_batch.device)
        input_ids_batch = torch.cat((input_ids_batch, padding), dim=1)
    if label_ids_batch.size(1) < max_len:
        padding = torch.full((label_ids_batch.size(0), max_len - label_ids_batch.size(1)), -100).to(label_ids_batch.device)
        label_ids_batch = torch.cat((label_ids_batch, padding), dim=1)

    return input_ids_batch, label_ids_batch


def post_train_model(model, tokenizer, train_loader, optimizer, scheduler, device):
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for step, (input_ids, label_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)

            # Обработаем максимальную длину батча
            max_len = max(input_ids.size(1), label_ids.size(1))
            if input_ids.size(1) < max_len:
                padding = torch.full((input_ids.size(0), max_len - input_ids.size(1)), tokenizer.pad_token_id).to(device)
                input_ids = torch.cat((input_ids, padding), dim=1)
            if label_ids.size(1) < max_len:
                padding = torch.full((label_ids.size(0), max_len - label_ids.size(1)), -100).to(device)
                label_ids = torch.cat((label_ids, padding), dim=1)

            model.zero_grad()
            outputs = model(input_ids, labels=label_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 10 == 0:  # Проверим каждые 10 шагов
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader)}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Trained model not found. Please train the model first.")
        return

    model.to(device)

    new_dialogues = load_new_data()
    new_dataset = NewDialogueDataset(tokenizer, new_dialogues)
    train_loader = DataLoader(new_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model = post_train_model(model, tokenizer, train_loader, optimizer, scheduler, device)

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model post-trained and saved successfully.")


if __name__ == "__main__":
    main()
