# NLPCreditCardQandA

This project explores the use of summarization and question answering techniques in NLP. It includes a pipeline for summarizing credit card agreements, training models (LSTM, PGN), and conducting experiments to evaluate the quality of summaries and question-answering models. The experiments compare the performance of LSTM, PGN, and GPT-based methods on key metrics like ROUGE and accuracy, and LLMs as a judge.

The Required Dependencies are listed in the requirement.txt file. A quick way to download all of them would be to run the following command:

pip install -r requirements.txt 

There is a main.py script that when run acts as our complete pipeline for the project. It will generate the abstractive summaries, train the LSTM and PGN models, run experiment 1, and run experiement 2.

If a user wants to test the summarizers or the question answering models separately, they can go to the respective files (end with a testing) and can put in their sample text in the spots that are listed in the code. 

However, experiment 1 cannot be run standalone until the user generates an API key in the Groq application to run the Ollama LLM. Our API Key was stored safely in the environment variables. 
