import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


def run_abstractive_summarization(
    input_file="processed_files.csv", output_file="summarized_data.csv"
):
    # Load the model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move model to MPS (Metal Performance Shaders) - To accelerate GPU in PyTorch
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model = model.to(device)

    # This method is used to generate abstractive summaries for each of the credit card documents.
    # Since these credit card documents are extremely long, we break the documents into chunks and generate the abstractive summaries on those chunks
    # and joins all of them together
    def summarize_large_text(
        text, row_index, chunk_size=512, max_length=150, min_length=50
    ):

        if not isinstance(text, str) or not text.strip():
            print(f"Row {row_index}: No content available.")
            return "No content available"

        # Split the text into chunks
        words = text.split()
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Row {row_index}, Chunk {i + 1}/{len(chunks)}: Summarizing...")
            try:
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True).to(
                    device
                )
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                )
                summaries.append(
                    tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                )
            except Exception as e:
                print(f"Row {row_index}, Chunk {i + 1}: Error summarizing - {e}")
                summaries.append("")

        # Combine summaries
        final_summary = " ".join(summaries)
        print(f"Row {row_index}: Summary complete.")
        return final_summary

    # Load your DataFrame
    df = pd.read_csv(input_file)

    # Summarize the Content column with progress tracking
    print("Starting summarization...")
    df["Summary"] = df["Content"].apply(
        lambda x: summarize_large_text(x, df[df["Content"] == x].index[0])
    )

    print("Summarization complete.")

    # Save the updated DataFrame
    df.to_csv(output_file, index=False)
    print("Saved summarized data to 'summarized_data.csv'.")
