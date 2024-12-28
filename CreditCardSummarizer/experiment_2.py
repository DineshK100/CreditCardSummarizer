import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BertTokenizer,
    BertForQuestionAnswering,
    pipeline,
)
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import pickle
from tqdm import tqdm
from collections import OrderedDict
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to saved models
VOCAB_PATH = "vocab.pkl"
PGN_SUMMARY_MODEL_PATH = "pgn_summarizer.pth"
PGN_QA_MODEL_PATH = "pgn_qa.pth"
LSTM_SUMMARY_MODEL_PATH = "lstm_summarizer.pth"
LSTM_QA_MODEL_PATH = "lstm_qa.pth"
RESULTS_PATH = "experiment_results.csv"

# Hyperparameters
MAX_INPUT_LENGTH = 1000
MAX_OUTPUT_LENGTH = 150
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3


# Load Vocabulary
def load_vocab():
    with open(VOCAB_PATH, "rb") as f:
        return pickle.load(f)


# Safely tokenize text
def tokenize_safe(text):
    if not isinstance(text, str):
        text = str(text)
    return word_tokenize(text)


# Collate function for padding (used in LSTM/PGN models)
def collate_fn(batch):
    contexts, questions, answers = zip(*batch)
    contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=0)
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=0)
    return contexts_padded, questions_padded, answers


# Load summarization and QA models
def load_pgn_summarizer(vocab):
    class PointerGeneratorSummarizer(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.encoder = torch.nn.LSTM(
                embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
            )
            self.decoder = torch.nn.LSTM(
                embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True
            )
            self.fc = torch.nn.Linear(hidden_dim, vocab_size)

        def forward(self, input_tensor):
            embedded = self.embedding(input_tensor)
            encoder_output, (hidden, cell) = self.encoder(embedded)
            return encoder_output, hidden, cell

    model = PointerGeneratorSummarizer(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(PGN_SUMMARY_MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_lstm_summarizer(vocab):
    class LSTMSummarizer(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.encoder = torch.nn.LSTM(
                embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
            )
            self.decoder = torch.nn.LSTM(
                embedding_dim, hidden_dim, num_layers, batch_first=True
            )
            self.fc = torch.nn.Linear(hidden_dim, vocab_size)

        def forward(self, input_tensor):
            embedded = self.embedding(input_tensor)
            encoder_output, (hidden, cell) = self.encoder(embedded)
            return encoder_output, hidden, cell

    model = LSTMSummarizer(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(LSTM_SUMMARY_MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_bart_summarizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(
        device
    )
    return model, tokenizer


def load_pgn_qa(vocab):
    class PointerGeneratorQA(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.encoder = torch.nn.LSTM(
                embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
            )
            self.decoder = torch.nn.LSTM(
                embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True
            )
            self.fc = torch.nn.Linear(hidden_dim, vocab_size)

        def forward(self, context, question):
            context_embedded = self.embedding(context)
            context_outputs, (hidden, cell) = self.encoder(context_embedded)
            return context_outputs, hidden, cell

    model = PointerGeneratorQA(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(PGN_QA_MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_lstm_qa(vocab):
    class LSTMQuestionAnsweringModel(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.encoder = torch.nn.LSTM(
                embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
            )
            self.fc_start = torch.nn.Linear(hidden_dim, 1)
            self.fc_end = torch.nn.Linear(hidden_dim, 1)

        def forward(self, context_tensor, question_tensor):
            context_embedded = self.embedding(context_tensor)
            context_output, _ = self.encoder(context_embedded)
            start_logits = self.fc_start(context_output).squeeze(-1)
            end_logits = self.fc_end(context_output).squeeze(-1)
            return start_logits, end_logits

    model = LSTMQuestionAnsweringModel(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(LSTM_QA_MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_bert_qa():
    tokenizer = BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    model = BertForQuestionAnswering.from_pretrained(
        "deepset/bert-base-cased-squad2"
    ).to(device)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline


# Experiment Execution
def run_experiment():
    # Load models
    vocab = load_vocab()
    pgn_summarizer = load_pgn_summarizer(vocab)
    lstm_summarizer = load_lstm_summarizer(vocab)
    bart_summarizer, bart_tokenizer = load_bart_summarizer()
    pgn_qa = load_pgn_qa(vocab)
    lstm_qa = load_lstm_qa(vocab)
    bert_qa = load_bert_qa()

    # Define combinations
    combinations = [
        ("Original Document", "LSTM"),
        ("Original Document", "PGN"),
        ("Original Document", "BERT"),
        ("BART", "LSTM"),
        ("BART", "PGN"),
        ("BART", "BERT"),
        ("LSTM", "LSTM"),
        ("LSTM", "PGN"),
        ("LSTM", "BERT"),
        ("PGN", "LSTM"),
        ("PGN", "PGN"),
        ("PGN", "BERT"),
    ]

    data = pd.read_csv("question_answering_testset.csv")
    results = []

    # Process combinations
    for summary_model_type, qa_model_type in tqdm(
        combinations, desc="Running experiments"
    ):
        for index, row in data.iterrows():
            original_context = row["Context"]
            question = row["Question"]
            true_answer = row["Answer"]

            # Summarization
            if summary_model_type == "Original Document":
                summarized_context = original_context
            elif summary_model_type == "BART":
                summarized_context = summarize_text(
                    bart_summarizer, bart_tokenizer, original_context, "BART"
                )
            elif summary_model_type == "LSTM":
                summarized_context = summarize_text(
                    lstm_summarizer, None, original_context, "LSTM", vocab
                )
            elif summary_model_type == "PGN":
                summarized_context = summarize_text(
                    pgn_summarizer, None, original_context, "PGN", vocab
                )

            start_time = time.time()
            # Question Answering
            if qa_model_type == "LSTM":
                predicted_answer = answer_question(
                    lstm_qa, summarized_context, question, "LSTM", vocab
                )
            elif qa_model_type == "PGN":
                predicted_answer = answer_question(
                    pgn_qa, summarized_context, question, "PGN", vocab
                )
            elif qa_model_type == "BERT":
                predicted_answer = answer_question(
                    bert_qa, summarized_context, question, "BERT"
                )
            else:
                predicted_answer = "No Answer"

            time_taken = time.time() - start_time
            # Append results
            results.append(
                {
                    "Summary Model": summary_model_type,
                    "QA Model": qa_model_type,
                    "Original Context": original_context,
                    "Summarized Context": summarized_context,
                    "Question": question,
                    "True Answer": true_answer,
                    "Predicted Answer": predicted_answer,
                    "Time Taken (seconds)": time_taken,
                }
            )

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Experiment completed. Results saved to {RESULTS_PATH}")


# Summarization function
def summarize_text(summary_model, tokenizer, text, model_type, vocab=None):
    if model_type == "PGN" or model_type == "LSTM":
        tokens = tokenize_safe(text)[:MAX_INPUT_LENGTH]
        input_tensor = torch.tensor(
            [vocab.get(word, vocab["<UNK>"]) for word in tokens], device=device
        ).unsqueeze(0)
        _, hidden, cell = summary_model(input_tensor)
        return " ".join(tokens)  # Simulate returning the input
    elif model_type == "BART":
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt").to(device)
        summary_ids = summary_model.generate(inputs, max_length=MAX_OUTPUT_LENGTH)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    else:
        return text


# Question answering function
def answer_question(qa_model, context, question, model_type, vocab=None):
    if model_type == "LSTM":
        context_tokens = tokenize_safe(context)[:MAX_INPUT_LENGTH]
        question_tokens = tokenize_safe(question)[:MAX_INPUT_LENGTH]
        context_ids = torch.tensor(
            [vocab.get(word, vocab["<UNK>"]) for word in context_tokens], device=device
        ).unsqueeze(0)
        question_ids = torch.tensor(
            [vocab.get(word, vocab["<UNK>"]) for word in question_tokens], device=device
        ).unsqueeze(0)
        with torch.no_grad():
            start_logits, end_logits = qa_model(context_ids, question_ids)
            start_idx = torch.argmax(start_logits, dim=1).item()
            end_idx = torch.argmax(end_logits, dim=1).item()
            tokens = context_tokens[start_idx : end_idx + 1]
        return " ".join(tokens) if tokens else "No Answer"
    elif model_type == "PGN":
        return "Answer from PGN QA (Simulated)"
    elif model_type == "BERT":
        return qa_model(question=question, context=context)["answer"]
    else:
        return "No Answer"


if __name__ == "__main__":
    run_experiment()
