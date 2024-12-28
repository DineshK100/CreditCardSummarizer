import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import pickle


def train_lstm_summarizer(data_file="golden_summaries.csv"):
    # Set device (if my computer's GPU is available use that otherwise use the CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters for the LSTM model
    MAX_INPUT_LENGTH = 1000
    MAX_OUTPUT_LENGTH = 150
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    EPOCHS = 15

    # Tokenizes input text into word-level tokens using NLTK's word_tokenize.
    def tokenize(text):
        return word_tokenize(text)

    # PyTorch Dataset for handling text summarization data.
    # Handles tokenization, truncation, and weight assignment for LLM-generated summaries.
    class WeightedTextSummaryDataset(Dataset):
        def __init__(self, data, max_input_length, max_output_length):
            self.data = data
            self.max_input_length = max_input_length
            self.max_output_length = max_output_length
            self.weights = (
                data["IsLLMGenerated"]
                .apply(
                    lambda x: (
                        2.0 if x == 1 else 1.0
                    )  # Assign higher weights to LLM summaries
                )
                .values
            )

        # Returns the number of samples in the dataset
        def __len__(self):
            return len(self.data)

        # Returns a single sample from the dataset, including the input tensor, target tensor, and weight.
        def __getitem__(self, idx):
            input_text = tokenize(self.data.iloc[idx]["Content"])[
                : self.max_input_length
            ]
            target_text = tokenize(self.data.iloc[idx]["TargetSummary"])[
                : self.max_output_length
            ]

            input_tensor = torch.tensor(
                [vocab.get(word, vocab["<UNK>"]) for word in input_text]
            )
            target_tensor = torch.tensor(
                [vocab.get(word, vocab["<UNK>"]) for word in target_text]
            )
            weight = self.weights[idx]

            return input_tensor, target_tensor, weight

    # Custom collate function for DataLoader to handle padding and batching with weights. Pads input and target sequences to the same length and bundles weights.
    def collate_fn_with_weights(batch):
        inputs, targets, weights = zip(*batch)
        inputs_padded = pad_sequence(
            inputs, batch_first=True, padding_value=vocab["<PAD>"]
        )
        targets_padded = pad_sequence(
            targets, batch_first=True, padding_value=vocab["<PAD>"]
        )
        weights = torch.tensor(weights, dtype=torch.float32)
        return inputs_padded, targets_padded, weights

    # LSTM-based sequence-to-sequence model for text summarization.
    class LSTMTextSummarizer(nn.Module):
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

        # Forward pass through the model. Encodes the input sequence and decodes the output sequence.
        def forward(self, input_tensor, target_tensor=None):
            embedded = self.embedding(input_tensor)
            encoder_output, (hidden, cell) = self.encoder(embedded)

            if target_tensor is not None:
                decoder_input = self.embedding(target_tensor[:, :-1])
                decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
                output = self.fc(decoder_output)
                return output
            else:
                return hidden, cell

    # Loads in the dataset and converts the necessary columns into the correct types for the model
    data = pd.read_csv(data_file)
    data["Content"] = data["Content"].astype(str)
    data["Summary"] = data["Summary"].astype(str)
    data["LLMSummaries"] = data["LLMSummaries"].fillna("")

    # If an LLM-generated summary exists, use it; otherwise, use the human-provided summary
    data["TargetSummary"] = data.apply(
        lambda row: row["LLMSummaries"] if row["LLMSummaries"] else row["Summary"],
        axis=1,
    )

    # Column used to check whether the summary is a golden standard summary generated by an LLM
    data["IsLLMGenerated"] = data["LLMSummaries"].apply(lambda x: 1 if x else 0)

    # Create the vocabulary
    # Combine all content and summaries into a single text string and tokenize it to create a unique vocabulary
    all_text = " ".join(data["Content"] + " " + data["Summary"])
    unique_tokens = list(set(tokenize(all_text)))
    vocab = {word: idx for idx, word in enumerate(unique_tokens, start=2)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    # Save the vocabulary to a pickle file for future use
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Split data into training and testing data
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)
    # Further split training data into 90% training and 10% validation
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # Create PyTorch datasets for training, validation, and testing
    train_dataset = WeightedTextSummaryDataset(
        train_data, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH
    )

    val_dataset = WeightedTextSummaryDataset(
        val_data, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH
    )

    test_dataset = WeightedTextSummaryDataset(
        test_data, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH
    )

    # Create DataLoaders for batching and shuffling data
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn_with_weights,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_weights
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_with_weights
    )

    # Initialize the LSTM model for text summarization
    model = LSTMTextSummarizer(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)

    # Define the loss function
    # Use cross-entropy loss, ignoring the padding index to avoid penalizing padded tokens
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

    # Define the optimizer
    # Use Adam optimizer with a predefined learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Trains the LSTM-based text summarization model. Includes training, validation, and model checkpoint saving after each epoch.
    def train_model():
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            for inputs, targets in tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}"
            ):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(inputs, targets)
                loss = criterion(
                    output.view(-1, len(vocab)), targets[:, 1:].reshape(-1)
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs, targets)
                    loss = criterion(
                        output.view(-1, len(vocab)), targets[:, 1:].reshape(-1)
                    )
                    val_loss += loss.item()
            print(f"Validation Loss: {val_loss / len(val_loader)}")

            # Save the model after every epoch
            torch.save(model.state_dict(), f"lstm_epoch_{epoch + 1}.pth")
            print(f"Model saved for Epoch {epoch + 1}")

        print("Training complete.")

    print("Training complete. Models saved for each epoch.")

    if __name__ == "__main__":
        train_model()
