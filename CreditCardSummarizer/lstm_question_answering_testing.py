import torch
import time
import torch.nn as nn
from nltk.tokenize import word_tokenize
import pickle


# Model definition
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


# Load vocabulary
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

reverse_vocab = {idx: word for word, idx in vocab.items()}

# Add missing tokens to vocabulary
for token in word_tokenize("The annual percentage rate APR maintaining"):
    if token not in vocab:
        vocab[token] = len(vocab)

# Initialize model
vocab_size = len(vocab)
EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT = 256, 512, 2, 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMQuestionAnsweringModel(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

# Load state dictionary from checkpoint
state_dict = torch.load("lstm_question_answering_model", map_location=device)

# Fix embedding layer size mismatch
loaded_vocab_size = state_dict["embedding.weight"].size(
    0
)  # Original vocab size in the checkpoint
current_vocab_size = len(vocab)  # Current vocab size

if current_vocab_size > loaded_vocab_size:
    # Randomly initialize embeddings for new tokens
    with torch.no_grad():
        extra_rows = torch.empty(
            (
                current_vocab_size - loaded_vocab_size,
                state_dict["embedding.weight"].size(1),
            ),
            device=device,
        )
        nn.init.uniform_(extra_rows, -0.1, 0.1)  # Initialize with small random values
        state_dict["embedding.weight"] = torch.cat(
            [state_dict["embedding.weight"], extra_rows], dim=0
        )
    print(
        f"Expanded embedding weights with random initialization to match current vocabulary size: {current_vocab_size}."
    )
elif current_vocab_size < loaded_vocab_size:
    # Truncate the embedding weights
    with torch.no_grad():
        state_dict["embedding.weight"] = state_dict["embedding.weight"][
            :current_vocab_size, :
        ]
    print(
        f"Truncated embedding weights to match current vocabulary size: {current_vocab_size}."
    )

# Load adjusted state dictionary into the model
model.load_state_dict(state_dict, strict=True)
model.eval()


# Tokenizer
def tokenize(text):
    return word_tokenize(text)


# Predict answer function
def predict_answer(model, context, question, vocab):
    context_tokens = tokenize(context)
    question_tokens = tokenize(question)

    print(f"Context Tokens: {context_tokens}")
    print(f"Question Tokens: {question_tokens}")

    context_ids = torch.tensor(
        [vocab.get(word, vocab["<UNK>"]) for word in context_tokens],
        device=device,
    ).unsqueeze(0)

    question_ids = torch.tensor(
        [vocab.get(word, vocab["<UNK>"]) for word in question_tokens],
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        start_logits, end_logits = model(context_ids, question_ids)

        start_probs = torch.softmax(start_logits, dim=1)
        end_probs = torch.softmax(end_logits, dim=1)

        start_idx = torch.argmax(start_probs, dim=1).item()
        end_idx = torch.argmax(end_probs, dim=1).item()

        print(f"Start Logits: {start_logits}")
        print(f"End Logits: {end_logits}")
        print(f"Start Probabilities: {start_probs}")
        print(f"End Probabilities: {end_probs}")
        print(f"Start Index: {start_idx}, End Index: {end_idx}")

        # Debugging top probabilities
        top_start_indices = torch.topk(start_probs, 3, dim=1)
        top_end_indices = torch.topk(end_probs, 3, dim=1)
        print(f"Top Start Indices and Probabilities: {top_start_indices}")
        print(f"Top End Indices and Probabilities: {top_end_indices}")

    if (
        start_idx < len(context_tokens)
        and end_idx < len(context_tokens)
        and start_idx <= end_idx
    ):
        predicted_answer = " ".join(context_tokens[start_idx : end_idx + 15])
    else:
        predicted_answer = "Answer span out of bounds."

    print(f"Predicted Start Token: {context_tokens[start_idx]} (Index: {start_idx})")
    print(f"Predicted End Token: {context_tokens[end_idx]} (Index: {end_idx})")

    return predicted_answer


# Test
context = """First Command Bank Visa Cardholder Agreement (Platinum/Classic) The person(s) ("Cardholder," whether one or more) who signed and returned the Application for a Visa ("Application") has requested First Command Bank ("Bank") to extend to Cardholder open-end credit. anyone else using the Card unless the use of such Card is by a person other than the Cardholder. Bank will inform Cardholder from time to time of the maximum amount of debt ("Credit Limit") that may be outstanding in the Account at any time. Cardholder agrees not to use the Card in any manner that would cause the outstanding balance to exceed the Credit Limit. Bank may designate that only a portion of Cardholder's Credit Limit is available for Cash Advances. and providing Cardholder information about products and services. As of the end of each monthly billing cycle, Cardholder will be furnished a periodic statement showing, among other things, (a) the amount owed ("Previous Balance") at the beginning of the billing cycle. If Cardholder is composed of more than one person, only one periodic statement will be provided. Charge, Bank begins to charge the Finance Charge on all amounts Cardholder owes Bank (except "New Purchases") from the first day of the billing cycle. A New Purchase is one that appears on the periodic statement for the first time. Bank calculates the Finance charge on the Account by applying the "Periodic Rate" (defined below) to the "Average Daily Balance" Bank adds 3% to the Prime Rate to determine Platinum APR (Periodic Rate currently .005208). For Credit Purchases and Cash Advances which occur on a Platinum Account, the APR varies with changes to the prime rate. All payments received on or before 5 o'clock p.m. (Fort Worth, Texas time) on Bank's business day at the address indicated on the periodic statement will be credited to the Account. Convenience Check, the instructions which Bank provides when the Convenience Checks are issued must be followed. Cardholder agrees to hold Bank harmless and indemnify Bank for any losses, expenses and costs, including attorney's fees incurred by Bank. Bank will use its best efforts to stop payment, but will incur no liability if it is unable to. then owed to Bank by Cardholder immediately due and payable, without prior notice or demand of any kind, except as may be required by applicable law. Bank may increase the Annual Percentage Rate up to 18%, which is the Default Rate under the Table of Charges. Cardholder agrees to pay all amounts actually incurred by Bank as court costs and attorneys' fees. not complete a transaction to or from the Account on time or in the correct amount according to this Agreement, Bank may be liable for Cardhold- er's losses or damages. Bank will not be liable if, through no fault of Bank's, the available credit is insufficient for the transaction or is unavailable for withdrawal. Cardholder authorizes Bank to share information about Cardholder's payment history with other persons or companies when permitted by law. Bank will not be responsible for merchandise or services purchased by Cardholder with the Card or Convenience Check unless required by law. Any refund, adjustment or credit allowed by a Seller shall not be cash, but rather be by a credit advice to Bank which shall be shown as a credit on the periodic statement. Bank is subject to the requirement of the USA Patriot Act. Bank may obtain at any time Cardholder's credit reports for any legitimate purpose associated with the Account or the application or request for an Account. Ohio anti-discrimination laws require creditors to make credit equally available to all creditworthy customers. Married Wisconsin Applicants: No provisions of any marital property agreement, unilateral statement, or court order applying to marital property will adversely affect a creditor's interests unless prior to the time credit is granted. on the goods or services. You have this protection only when the purchase price was more than $50 and the purchase was made in your home state. (If we own or operate the merchant, or if we mailed the advertisement for the property or services, all pur- chases are covered regardless of amount or location of purchase.)"""
question = "What is the APR rate?"

startTime = time.time()
predicted_answer = predict_answer(model, context, question, vocab)
endTime = time.time()

print(f"Question: {question}")
print(f"Predicted Answer: {predicted_answer}")
print((endTime - startTime) * 1000)
