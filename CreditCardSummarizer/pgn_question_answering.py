import json
import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from collections import Counter
import nltk

# nltk.download("punkt_tab")

# Hyperparameters for the PGN Model for Question Answering
MAX_INPUT_LENGTH = 1000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.0001
MAX_OUTPUT_LENGTH = 1500
BATCH_SIZE = 8
EPOCHS = 30

# Set the device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Safely tokenizes the input text into words.
# Converts non-string input to string before tokenizing.
def tokenize_safe(text):
    if not isinstance(text, str):
        text = str(text)
    return word_tokenize(text)


# Preprocesses the dataset by loading JSON data and combining fields into context, question, and answer.
def preprocess_dataset(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    processed_data = []
    for entry in data:
        # Combine pre_text and post_text as context
        pre_text = entry.get("pre_text", "")
        post_text = entry.get("post_text", "")
        context = " ".join(pre_text) + " " + " ".join(post_text)

        # Extract question and answer
        qa = entry.get("qa", {})
        question = qa.get("question", "")
        answer = qa.get("exe_ans", "")

        # Ensure context, question, and answer are strings
        context = str(context)
        question = str(question)
        answer = str(answer)

        processed_data.append(
            {"context": context, "question": question, "answer": answer}
        )

    return pd.DataFrame(processed_data)


# Builds a vocabulary from the dataset with special tokens.
def build_vocab(data, special_tokens):
    counter = Counter()
    for _, row in data.iterrows():
        # Tokenize and count words in context, question, and answer
        counter.update(tokenize_safe(row["context"]))
        counter.update(tokenize_safe(row["question"]))
        counter.update(tokenize_safe(row["answer"]))

    # Create vocab dictionary with special tokens
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for word, _ in counter.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM-based encoder for processing input sequences.
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )

    # Forward pass through the encoder.
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)
        return output, hidden


from torch.nn.utils.rnn import pad_sequence


# Custom collate function to pad sequences to the same length for batch processing.
def collate_fn(batch):
    contexts, questions, answers = zip(*batch)

    # Pad sequences to the same length
    contexts_padded = pad_sequence(
        contexts, batch_first=True, padding_value=vocab["<PAD>"]
    )
    questions_padded = pad_sequence(
        questions, batch_first=True, padding_value=vocab["<PAD>"]
    )
    answers_padded = pad_sequence(
        answers, batch_first=True, padding_value=vocab["<PAD>"]
    )

    return contexts_padded, questions_padded, answers_padded


# Attention mechanism for aligning decoder with relevant encoder outputs.
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    # Computes attention weights over encoder outputs.
    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1]  # Use the last layer of hidden state
        hidden_expanded = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = F.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
        attention_weights = F.softmax(self.v(energy).squeeze(-1), dim=1)
        return attention_weights


# Pointer-Generator Network for text summarization and question answering.
class PointerGeneratorNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(PointerGeneratorNetwork, self).__init__()
        self.encoder = Encoder(
            vocab_size, embedding_dim, hidden_dim, num_layers, dropout
        )
        self.decoder = nn.LSTM(
            embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.attention = Attention(hidden_dim)
        self.generator = nn.Linear(hidden_dim, vocab_size)
        self.pointer = nn.Linear(hidden_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    # Forward pass for the Pointer-Generator Network.
    def forward(self, context, question, answer):
        context_outputs, context_hidden = self.encoder(context)
        question_outputs, question_hidden = self.encoder(question)

        decoder_hidden = question_hidden

        start_token = torch.full(
            (context.size(0), 1),
            vocab["<START>"],
            device=context.device,
            dtype=torch.long,
        )
        decoder_input = self.embedding(start_token)

        outputs = []
        for t in range(answer.size(1)):
            attn_weights = self.attention(decoder_hidden[0], context_outputs)
            context_vector = torch.bmm(attn_weights.unsqueeze(1), context_outputs)

            lstm_input = torch.cat((decoder_input, context_vector), dim=2)

            decoder_output, decoder_hidden = self.decoder(lstm_input, decoder_hidden)

            vocab_dist = F.softmax(self.generator(decoder_output.squeeze(1)), dim=1)

            pointer_dist = attn_weights

            final_dist = vocab_dist.clone()
            for i in range(context.size(0)):
                for j in range(context.size(1)):
                    token_id = context[i, j].item()
                    if token_id != vocab["<PAD>"]:
                        final_dist[i, token_id] += pointer_dist[i, j]

            outputs.append(final_dist.unsqueeze(1))

            if t < answer.size(1) - 1:
                decoder_input = self.embedding(answer[:, t].unsqueeze(1))

        return torch.cat(outputs, dim=1)


from torch.utils.data import Dataset, DataLoader


# Dataset class for handling question answering data.
class QADataset(Dataset):
    def __init__(self, dataframe, vocab, max_len):
        self.data = dataframe
        self.vocab = vocab
        self.max_len = max_len

    # Returns the total number of samples in the dataset.
    def __len__(self):
        return len(self.data)

    # Returns a single sample from the dataset.

    def __getitem__(self, idx):
        context = str(self.data.iloc[idx]["context"])
        question = str(self.data.iloc[idx]["question"])
        answer = str(self.data.iloc[idx]["answer"])

        context_ids = [
            self.vocab.get(w, self.vocab["<UNK>"]) for w in tokenize_safe(context)
        ][: self.max_len]
        question_ids = [
            self.vocab.get(w, self.vocab["<UNK>"]) for w in tokenize_safe(question)
        ][: self.max_len]
        answer_ids = [self.vocab["<START>"]] + [
            self.vocab.get(w, self.vocab["<UNK>"]) for w in tokenize_safe(answer)
        ][: self.max_len - 1]

        return (
            torch.tensor(context_ids, dtype=torch.long),
            torch.tensor(question_ids, dtype=torch.long),
            torch.tensor(answer_ids, dtype=torch.long),
        )


# Load train and test data
train_data = preprocess_dataset("dataset/train.json")
test_data = preprocess_dataset("dataset/test.json")

# Define special tokens
special_tokens = ["<PAD>", "<START>", "<UNK>"]

# Build vocab from train_data
vocab = build_vocab(train_data, special_tokens)
for token in special_tokens:
    if token not in vocab:
        vocab[token] = len(vocab)


def train_pgn_qa(train_file, test_file):
    # Load train and test data
    train_data = preprocess_dataset(train_file)
    test_data = preprocess_dataset(test_file)

    # Define special tokens
    special_tokens = ["<PAD>", "<START>", "<UNK>"]

    # Build vocab from train_data
    vocab = build_vocab(train_data, special_tokens)
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    # Create dataset and dataloader
    train_dataset = QADataset(train_data, vocab, MAX_INPUT_LENGTH)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    model = PointerGeneratorNetwork(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    import os

    # Create a directory to save the models
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Training loop with model saving
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for context, question, answer in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            context, question, answer = (
                context.to(device),
                question.to(device),
                answer.to(device),
            )

            optimizer.zero_grad()
            outputs = model(context, question, answer)
            loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader)}")

        # Save the model
        save_path = os.path.join(save_dir, f"pointer_generator_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_pgn_qa("dataset/train.json", "dataset/test.json")
