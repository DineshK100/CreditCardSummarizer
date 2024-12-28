import torch
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from pgn_summarization import (
    build_vocab,
    Encoder,
    PointerGeneratorDecoder,
    PointerGenerator,
    collate_fn,
)


# Loads a pre-trained Pointer-Generator model from a checkpoint file.
def load_trained_model(vocab_size, emb_dim, hid_dim, checkpoint_path, device):
    encoder = Encoder(vocab_size, emb_dim, hid_dim).to(device)
    decoder = PointerGeneratorDecoder(vocab_size, emb_dim, hid_dim).to(device)
    model = PointerGenerator(encoder, decoder, vocab_size).to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Trained model loaded successfully.")

    return model


# Tokenizes and converts input text into a tensor of indices for the model.
def preprocess_input(text, vocab):
    tokens = word_tokenize(text.lower())
    indices = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    return torch.tensor(indices).unsqueeze(0)  # Add batch dimension


# Generates a summary for the given input using the trained model.
def generate_summary(model, input_tensor, vocab, max_len=50, device="cpu", top_k=5):
    model.eval()
    with torch.no_grad():
        src = input_tensor.to(device)

        encoder_outputs, hidden, cell = model.encoder(src)

        trg_vocab_inv = {v: k for k, v in vocab.items()}  # Invert vocab for decoding
        trg = torch.tensor([vocab["<PAD>"]]).to(device)
        summary = []

        for _ in range(max_len):
            output, _, hidden, cell = model.decoder(
                trg, hidden, cell, encoder_outputs, src
            )
            probabilities = torch.softmax(output, dim=1)

            # Apply top-k sampling
            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=1)
                top_k_probs = top_k_probs.squeeze(0)
                top_k_indices = top_k_indices.squeeze(0)
                chosen_index = torch.multinomial(top_k_probs, 1).item()
                word_idx = top_k_indices[chosen_index].item()
            else:
                word_idx = probabilities.argmax(1).item()

            word = trg_vocab_inv[word_idx]
            if word == "<PAD>":
                break
            summary.append(word)
            trg = torch.tensor([word_idx]).to(device)

        return " ".join(summary)


def getSummary(sample_text):
    # Load the dataset to extract the vocabulary
    data_path = "summarized_data.csv"  # Path to your dataset
    data = pd.read_csv(data_path)

    # Ensure the dataset has `content` and `summary` columns
    assert (
        "Content" in data.columns and "Summary" in data.columns
    ), "The dataset must have `content` and `summary` columns."

    # Hyperparameters
    EMB_DIM = 128
    HID_DIM = 256
    CHECKPOINT_PATH = "model_checkpoint.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabulary
    vocab = build_vocab(data)

    # Load trained model
    model = load_trained_model(len(vocab), EMB_DIM, HID_DIM, CHECKPOINT_PATH, DEVICE)

    # Test the model with a sample input
    sample_text = sample_text

    # Preprocess the input text
    input_tensor = preprocess_input(sample_text, vocab)

    # Generate summary
    summary = generate_summary(model, input_tensor, vocab, device=DEVICE)
    print("Generated Summary:")
    print(summary)


if __name__ == "__main__":
    # Load the dataset to extract the vocabulary
    data_path = "summarized_data.csv"  # Path to your dataset
    data = pd.read_csv(data_path)

    # Ensure the dataset has `content` and `summary` columns
    assert (
        "Content" in data.columns and "Summary" in data.columns
    ), "The dataset must have `content` and `summary` columns."

    # Hyperparameters
    EMB_DIM = 128
    HID_DIM = 256
    CHECKPOINT_PATH = "model_checkpoint.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build vocabulary
    vocab = build_vocab(data)

    # Load trained model
    model = load_trained_model(len(vocab), EMB_DIM, HID_DIM, CHECKPOINT_PATH, DEVICE)

    # Test the model with a sample input
    sample_text = "INPUT TEXT TO BE SUMMARIZED GOES HERE"

    # Preprocess the input text
    input_tensor = preprocess_input(sample_text, vocab)

    # Generate summary
    summary = generate_summary(model, input_tensor, vocab, device=DEVICE)
    print("Generated Summary:")
    print(summary)
