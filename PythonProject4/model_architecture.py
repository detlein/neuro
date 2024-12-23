import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

CLEANED_DATA_PATH = 'cleaned_dialogues.json'

def load_model_and_tokenizer(model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

def prepare_data(tokenizer, file_path=CLEANED_DATA_PATH):
    with open(file_path, 'r') as file:
        dialogues = json.load(file)

    inputs = []
    labels = []

    for dialogue in dialogues:
        input_ids = tokenizer.encode(dialogue['user_input'], add_special_tokens=True)
        label_ids = tokenizer.encode(dialogue['bot_response'], add_special_tokens=True)
        inputs.append(torch.tensor(input_ids))
        labels.append(torch.tensor(label_ids))

    return inputs, labels

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer('gpt2')
    inputs, labels = prepare_data(tokenizer)
    print(f"Loaded {len(inputs)} dialogues for training.")
