import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import pandas as pd
import json
import pickle
from tqdm import tqdm

# Set device for computation (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for the QA model
MAX_INPUT_LENGTH = 1000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.0001
MAX_OUTPUT_LENGTH = 1500
BATCH_SIZE = 8
EPOCHS = 30

# Load vocabulary from a pickle file
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Reverse vocabulary for converting token indices back to words
reverse_vocab = {idx: word for word, idx in vocab.items()}


# Tokenizes the input text into words using NLTK's word_tokenize.
def tokenize(text):
    return word_tokenize(text)


# Custom PyTorch Dataset to handle question-answering data.
class QAWithQuestionsDataset(Dataset):
    def __init__(self, data, max_input_length, vocab):
        self.data = data
        self.max_input_length = max_input_length
        self.vocab = vocab

    # Returns the total number of samples in the dataset.
    def __len__(self):
        return len(self.data)

    # Retrieves a single sample from the dataset.
    def __getitem__(self, idx):
        content = tokenize(self.data.iloc[idx]["Summary"])[: self.max_input_length]
        question = tokenize(self.data.iloc[idx]["Question"])[: self.max_input_length]
        answer = tokenize(self.data.iloc[idx]["Answer"])[:MAX_OUTPUT_LENGTH]

        # Convert text to indices
        content_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in content]
        question_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in question]

        # Find start and end spans in the context
        content_str = " ".join(content)
        answer_str = " ".join(answer)

        # Default values if the answer cannot be found
        start_idx, end_idx = 0, 0

        if answer_str in content_str:
            start_idx = content_str.find(answer_str)
            start_idx = len(tokenize(content_str[:start_idx]))
            end_idx = start_idx + len(answer) - 1

        # If indices are out of bounds, clip them
        if start_idx < 0 or start_idx >= len(content):
            start_idx = 0
        if end_idx < 0 or end_idx >= len(content):
            end_idx = 0

        return (
            torch.tensor(content_ids),
            torch.tensor(question_ids),
            start_idx,
            end_idx,
        )


# Custom collate function for DataLoader to pad sequences to the same length.
def collate_fn(batch):
    contexts, questions, start_indices, end_indices = zip(*batch)
    contexts_padded = nn.utils.rnn.pad_sequence(
        contexts, batch_first=True, padding_value=vocab["<PAD>"]
    )
    questions_padded = nn.utils.rnn.pad_sequence(
        questions, batch_first=True, padding_value=vocab["<PAD>"]
    )
    return (
        contexts_padded,
        questions_padded,
        torch.tensor(start_indices),
        torch.tensor(end_indices),
    )


# LSTM-based Question Answering model for predicting start and end indices of answers.
class LSTMQuestionAnsweringModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMQuestionAnsweringModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=vocab["<PAD>"]
        )
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.fc_start = nn.Linear(hidden_dim, 1)  # For predicting start logits
        self.fc_end = nn.Linear(hidden_dim, 1)  # For predicting end logits

    #  Forward pass through the model.
    def forward(self, context_tensor, question_tensor):
        # Embed and encode context
        context_embedded = self.embedding(context_tensor)
        context_output, _ = self.encoder(context_embedded)

        # Embed and encode question
        question_embedded = self.embedding(question_tensor)
        question_output, _ = self.encoder(question_embedded)

        # Compute logits for start and end indices (based only on context)
        start_logits = self.fc_start(context_output).squeeze(-1)
        end_logits = self.fc_end(context_output).squeeze(-1)

        return start_logits, end_logits


# Converts JSON data into a DataFrame with 'Summary', 'Question', and 'Answer' columns.
# This is from the SQuAD dataset (there was already data split into test and train sets)
def json_to_dataframe(json_data):
    data_list = []
    for entry in json_data:
        # Ensure pre_text and post_text are strings
        pre_text = entry.get("pre_text", "")
        if isinstance(pre_text, list):
            pre_text = " ".join(pre_text)
        if not isinstance(pre_text, str):
            pre_text = ""

        post_text = entry.get("post_text", "")
        if isinstance(post_text, list):
            post_text = " ".join(post_text)
        if not isinstance(post_text, str):
            post_text = ""

        # Combine pre_text and post_text as "Summary"
        summary = pre_text + " " + post_text

        qa_data = entry.get("qa", {})
        question = qa_data.get("question", "")
        if not isinstance(question, str):
            question = ""

        answer = qa_data.get("exe_ans", "")
        if not isinstance(answer, str):
            answer = ""

        data_list.append({"Summary": summary, "Question": question, "Answer": answer})
    return pd.DataFrame(data_list)


def train_lstm_question_answering(
    train_file_path="dataset/train.json", test_file_path="dataset/test.json"
):
    # Paths to your train and test JSON files
    # train_file_path = "dataset/train.json"
    # test_file_path = "dataset/test.json"

    # Load and process train and test JSON files
    with open(train_file_path, "r") as train_file:
        train_json = json.load(train_file)

    with open(test_file_path, "r") as test_file:
        test_json = json.load(test_file)

    train_data = json_to_dataframe(train_json)
    test_data = json_to_dataframe(test_json)

    # Create the dataset objects
    train_dataset = QAWithQuestionsDataset(train_data, MAX_INPUT_LENGTH, vocab)
    test_dataset = QAWithQuestionsDataset(test_data, MAX_INPUT_LENGTH, vocab)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Initialize model
    model = LSTMQuestionAnsweringModel(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for contexts, questions, start_indices, end_indices in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}"
        ):
            contexts, questions = contexts.to(device), questions.to(device)
            start_indices, end_indices = start_indices.to(device), end_indices.to(
                device
            )

            optimizer.zero_grad()
            start_logits, end_logits = model(contexts, questions)
            loss_start = criterion(start_logits, start_indices)
            loss_end = criterion(end_logits, end_indices)
            loss = loss_start + loss_end
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

    # Save model
    torch.save(model.state_dict(), "qa_model_finbert.pth")
    print("Model training complete. Model saved as 'qa_model_finbert.pth'.")

    # Function to predict answers
    def predict_answer(model, context, question, vocab):
        context_ids = torch.tensor(
            [
                vocab.get(word, vocab["<UNK>"])
                for word in tokenize(context)[:MAX_INPUT_LENGTH]
            ],
            device=device,
        ).unsqueeze(0)
        question_ids = torch.tensor(
            [
                vocab.get(word, vocab["<UNK>"])
                for word in tokenize(question)[:MAX_INPUT_LENGTH]
            ],
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            start_logits, end_logits = model(context_ids, question_ids)

            # Apply softmax to convert logits into probabilities
            start_probs = torch.softmax(start_logits, dim=1)
            end_probs = torch.softmax(end_logits, dim=1)

            # Find start and end indices
            start_idx = torch.argmax(start_probs, dim=1).item()
            end_idx = torch.argmax(end_probs, dim=1).item()

        tokens = tokenize(context)
        if start_idx >= len(tokens) or end_idx >= len(tokens) or start_idx > end_idx:
            return "Answer span out of bounds."

        return " ".join(tokens[start_idx : end_idx + 1])


# Example Prediction
# sample_context = "Some sample context text here."
# sample_question = "Sample question?"
# predicted_answer = predict_answer(model, sample_context, sample_question, vocab)
# print(f"Question: {sample_question}")
# print(f"Predicted Answer: {predicted_answer}")
