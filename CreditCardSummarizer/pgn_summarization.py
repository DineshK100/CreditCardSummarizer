import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import nltk
import os
import random

# nltk.download("punkt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom PyTorch Dataset to handle credit card contract summarization data.
# Converts text into token indices using a given vocabulary.
class CreditCardDataset(Dataset):
    def __init__(self, data, vocab):
        self.contents = data["Content"].tolist()
        self.summaries = data["Summary"].tolist()
        self.vocab = vocab

    # Returns the total number of samples in the dataset.
    def __len__(self):
        return len(self.contents)

    # Returns tokenized and indexed document and summary for the given index.
    def __getitem__(self, idx):
        doc = word_tokenize(self.contents[idx].lower())
        summary = word_tokenize(self.summaries[idx].lower())

        doc_indices = [self.vocab.get(word, self.vocab["<UNK>"]) for word in doc]
        summary_indices = [
            self.vocab.get(word, self.vocab["<UNK>"]) for word in summary
        ]

        return torch.tensor(doc_indices), torch.tensor(summary_indices)


# Builds a vocabulary of the most common words from the dataset.
def build_vocab(data, vocab_size=5000):
    tokens = []
    for content, summary in zip(data["Content"], data["Summary"]):
        tokens.extend(word_tokenize(content.lower()))
        tokens.extend(word_tokenize(summary.lower()))
    most_common = Counter(tokens).most_common(vocab_size - 2)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


# Encoder that processes input sequences and generates context vectors.
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    # Processes the input sequence through the embedding layer and LSTM.
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell


# Custom collate function to pad sequences to the same length within a batch.
def collate_fn(batch):
    docs, summaries = zip(*batch)
    docs_padded = pad_sequence(docs, batch_first=True, padding_value=0)
    summaries_padded = pad_sequence(summaries, batch_first=True, padding_value=0)
    return docs_padded, summaries_padded


# LSTM-based Decoder with a Pointer-Generator mechanism for handling out-of-vocabulary words.
class PointerGeneratorDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hid_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.pointer = nn.Linear(hid_dim, 1)

    # Processes the current input and context using LSTM and generates predictions.
    def forward(self, input, hidden, cell, encoder_outputs, src):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)

        hidden_for_attention = hidden[0]
        if hidden_for_attention.dim() == 3:
            hidden_for_attention = hidden_for_attention.squeeze(0)
        hidden_for_attention = hidden_for_attention.unsqueeze(2)

        # Compute attention weights
        attn_weights = torch.bmm(encoder_outputs, hidden_for_attention).squeeze(2)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        # Combine context vector and embedded input
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # Generate predictions
        prediction = self.fc_out(output.squeeze(1))
        pointer_weights = torch.sigmoid(self.pointer(output.squeeze(1)))

        return prediction, pointer_weights, hidden, cell


# Pointer-Generator model combining Encoder and Decoder.
class PointerGenerator(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size

    # Performs forward pass through the model using teacher forcing for training.
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden, cell = self.encoder(src)
        trg_len = trg.shape[1]
        batch_size = trg.shape[0]
        outputs = torch.zeros(batch_size, trg_len, self.vocab_size).to(trg.device)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, _, hidden, cell = self.decoder(
                input, hidden, cell, encoder_outputs, src
            )
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs


# Trains the model for one epoch.
def train_model(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch_idx, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(src, trg)

        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        # Compute loss
        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        print(f"Batch {batch_idx + 1}/{len(iterator)}: Loss = {loss.item():.4f}")

    return epoch_loss / len(iterator)


# Evaluates the model on a validation or test set.
def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


import os


# Save and Load Model Functions
def save_model(model, optimizer, epoch, path="model_checkpoint.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


# Function to load the model
def load_model(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Model loaded from {path}, starting at epoch {start_epoch + 1}")
        return start_epoch
    else:
        print(f"No checkpoint found at {path}")
        return 0


def train_pgn(data_path="summarized_data.csv"):
    # Load your dataset
    # data_path = "summarized_data.csv"
    data = pd.read_csv(data_path)

    # Ensure the dataset has `content` and `summary` columns
    assert (
        "Content" in data.columns and "Summary" in data.columns
    ), "The dataset must have `content` and `summary` columns."

    # Hyperparameters
    vocab = build_vocab(data)
    BATCH_SIZE = 16  # Adjust batch size based on dataset size
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    EMB_DIM = 128
    HID_DIM = 256
    N_EPOCHS = 5
    CLIP = 1
    CHECKPOINT_PATH = "model_checkpoint.pth"

    # Dataset and Dataloader
    dataset = CreditCardDataset(data, vocab)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # Model
    encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM).to(device)
    decoder = PointerGeneratorDecoder(OUTPUT_DIM, EMB_DIM, HID_DIM).to(device)
    model = PointerGenerator(encoder, decoder, len(vocab)).to(device)

    # Optimizer and Criterion
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

    # Load the model if a checkpoint exists
    start_epoch = load_model(CHECKPOINT_PATH, model, optimizer)

    # Training
    for epoch in range(start_epoch, N_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{N_EPOCHS} training started...")
        train_loss = train_model(model, dataloader, optimizer, criterion, CLIP)
        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} completed. Training Loss: {train_loss:.3f}\n"
        )

        # Save the model at the end of each epoch
        save_model(model, optimizer, epoch, CHECKPOINT_PATH)
