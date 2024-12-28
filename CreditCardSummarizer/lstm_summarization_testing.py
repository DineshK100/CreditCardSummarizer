import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle
from lstm_summarization import tokenize
import numpy as np

# Set device for computation (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
MAX_INPUT_LENGTH = 1000
MAX_OUTPUT_LENGTH = 150
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

# Load vocab from the file system
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Define reverse vocab for converting indices back to words
reverse_vocab = {idx: word for word, idx in vocab.items()}


# Define the LSTM-based Text Summarization Model
class LSTMTextSummarizer(nn.Module):

    # Sequence-to-sequence LSTM model for text summarization.
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMTextSummarizer, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=vocab["<PAD>"]
        )
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True
        )
        self.decoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    # Forward pass for training or inference.
    def forward(self, input_tensor, target_tensor=None):
        embedded = self.embedding(input_tensor)
        encoder_output, (hidden, cell) = self.encoder(embedded)

        if target_tensor is not None:  # Training
            decoder_input = self.embedding(target_tensor[:, :-1])
            decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            return output
        else:  # Inference
            return hidden, cell


# Loads the trained model from disk and prepares it for inference.
def load_model():
    model = LSTMTextSummarizer(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)
    model.load_state_dict(
        torch.load("lstm_epoch_15.pth", map_location=device)
    )  # Load trained model weights
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model


# Generates a summary for the given text using the trained model.
def summarize_text(model, text, vocab, max_output_length=MAX_OUTPUT_LENGTH):
    print("Original Input Text:", text)

    tokens = tokenize(text)[:MAX_INPUT_LENGTH]
    print("Tokenized Input:", tokens)

    input_tensor = torch.tensor(
        [vocab.get(word, vocab["<UNK>"]) for word in tokens], device=device
    ).unsqueeze(0)
    print("Input Indices:", input_tensor)

    with torch.no_grad():
        _, (hidden, cell) = model.encoder(model.embedding(input_tensor))

    decoder_input = torch.tensor([[vocab["<PAD>"]]], device=device)
    generated_summary = []

    for i in range(max_output_length):
        with torch.no_grad():
            embedded = model.embedding(decoder_input)
            output, (hidden, cell) = model.decoder(embedded, (hidden, cell))
            logits = model.fc(output.squeeze(1))

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

            # Sample the next token probabilistically
            next_token = np.random.choice(len(probs), p=probs)

            print(
                f"Step {i+1}: Decoder Input: {decoder_input.item()}, "
                f"Next Token: {next_token} (Probability: {probs[next_token]:.4f})"
            )

            generated_summary.append(next_token)

            if next_token == vocab["<PAD>"]:
                break

            decoder_input = torch.tensor([[next_token]], device=device)

    summary_words = [reverse_vocab.get(idx, "<UNK>") for idx in generated_summary]
    print("Generated Summary Tokens:", summary_words)

    # Join words to form the final summary
    return " ".join(summary_words)


def getSummary(input_text):
    # Load the model
    model = load_model()

    # Input text for summarization
    input_text = input_text

    print("Starting Summary Generation...")

    summary = summarize_text(model, input_text, vocab)

    return summary


# Main script for summarization
if __name__ == "__main__":
    # Load the model
    model = load_model()

    # Input text for summarization
    input_text = """""SAMPLE CONTENT GOES HERE"""

    print("Starting Summary Generation...")

    summary = summarize_text(model, input_text, vocab)

    print("Generated Summary:")
    print(summary)
