from rouge import Rouge
import pandas as pd


def evaluate_rouge(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(input_file)

    # Initialize ROUGE scorer
    rouge = Rouge()

    # Columns to compare
    content_column = "Content"
    summary_columns = ["Summary", "LSTM Summaries", "PGN Summaries"]

    # Store results
    results = []

    for _, row in data.iterrows():
        content = str(row[content_column])

        for summary_column in summary_columns:
            summary = str(row[summary_column])
            scores = rouge.get_scores(summary, content, avg=True)

            results.append(
                {
                    "File Name": row.get("File Name", "N/A"),
                    "Summary Type": summary_column,
                    "ROUGE-1 Recall": scores["rouge-1"]["r"],
                    "ROUGE-1 Precision": scores["rouge-1"]["p"],
                    "ROUGE-1 F1": scores["rouge-1"]["f"],
                    "ROUGE-2 Recall": scores["rouge-2"]["r"],
                    "ROUGE-2 Precision": scores["rouge-2"]["p"],
                    "ROUGE-2 F1": scores["rouge-2"]["f"],
                    "ROUGE-L Recall": scores["rouge-l"]["r"],
                    "ROUGE-L Precision": scores["rouge-l"]["p"],
                    "ROUGE-L F1": scores["rouge-l"]["f"],
                }
            )

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"ROUGE scores saved to {output_file}")
