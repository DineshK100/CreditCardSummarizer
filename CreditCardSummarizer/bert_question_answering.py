from transformers import BertTokenizer, BertForQuestionAnswering, pipeline

# Load the BERT tokenizer and model fine-tuned on SQuAD
model_name = "deepset/bert-base-cased-squad2"  # BERT fine-tuned for question answering
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Example context and question
context = """SUMMARIZED TEXT GOES HERE"""

question = "QUESTION GOES HERE"

# Use the pipeline for QA
result = qa_pipeline(question=question, context=context)

# Output the result
print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print(f"Score: {result['score']}")
