import torch
from lstm_summarization_testing import (
    load_model as load_lstm_model,
    summarize_text,
    getSummary as getLstmSummary,
)
from pgn_summarization_testing import (
    load_trained_model,
    preprocess_input,
    generate_summary,
    getSummary,
)
import pandas as pd
import pickle

# Load vocab for both models
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)


def generate_summaries(input_file, output_file):
    # Load data
    data = pd.read_csv(input_file)

    lstm_summaries = []
    pgn_summaries = []

    # Generate summaries for each content
    for content in data["Content"]:
        lstm_summary = getLstmSummary(content)
        lstm_summaries.append(lstm_summary)

        input_tensor = preprocess_input(content, vocab)
        pgn_summary = getSummary(content)
        pgn_summaries.append(pgn_summary)

    # Add summaries to the DataFrame
    data["LSTM Summaries"] = lstm_summaries
    data["PGN Summaries"] = pgn_summaries

    # Save to output file
    data.to_csv(output_file, index=False)
    print(f"Summaries saved to {output_file}")


if __name__ == "__main__":
    generate_summaries(
        input_file="input_data.csv",
        lstm_model_path="lstm_epoch_15.pth",
        pgn_model_path="model_checkpoint.pth",
        output_file="output.csv",
    )
