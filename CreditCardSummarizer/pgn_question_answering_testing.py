import torch
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import pickle
from collections import OrderedDict

# Hyperparameters
MAX_INPUT_LENGTH = 1000
MAX_OUTPUT_LENGTH = 150
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Safely tokenizes the input text into words.
# Converts non-string input to string before tokenizing.
def tokenize_safe(text):
    if not isinstance(text, str):
        text = str(text)
    return word_tokenize(text)


# Loads vocabulary from a pickle file.
def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        return pickle.load(f)


# Pointer-Generator Network for question answering.
class PointerGeneratorNetwork(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(PointerGeneratorNetwork, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.encoder = torch.nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        self.decoder = torch.nn.LSTM(
            embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.Tanh(),
        )
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    # Forward pass for the model.
    def forward(self, context, question):
        context_embedded = self.embedding(context)
        context_outputs, (hidden, cell) = self.encoder(context_embedded)
        return context_outputs, (hidden, cell)


# Generates an answer based on the input context and question using the trained model.
def generate_answer(model, context_text, question_text, vocab):
    idx_to_word = {idx: word for word, idx in vocab.items()}

    context_tokens = tokenize_safe(context_text)[:MAX_INPUT_LENGTH]
    question_tokens = tokenize_safe(question_text)[:MAX_INPUT_LENGTH]
    context_ids = torch.tensor(
        [vocab.get(word, vocab["<UNK>"]) for word in context_tokens],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    question_ids = torch.tensor(
        [vocab.get(word, vocab["<UNK>"]) for word in question_tokens],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        context_outputs, (hidden, cell) = model(context_ids, question_ids)
        decoder_input = torch.tensor([[vocab["<START>"]]], device=device)
        generated_answer = []

        for _ in range(MAX_OUTPUT_LENGTH):
            decoder_embedded = model.embedding(decoder_input)
            context_vector = torch.bmm(
                torch.softmax(
                    torch.bmm(
                        hidden[-1].unsqueeze(1), context_outputs.permute(0, 2, 1)
                    ),
                    dim=-1,
                ),
                context_outputs,
            )
            lstm_input = torch.cat((decoder_embedded, context_vector), dim=2)
            decoder_output, (hidden, cell) = model.decoder(lstm_input, (hidden, cell))
            logits = model.fc(decoder_output.squeeze(1))
            probabilities = torch.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probabilities, 1).item()

            if predicted_token == vocab["<PAD>"]:
                break

            generated_answer.append(predicted_token)
            decoder_input = torch.tensor([[predicted_token]], device=device)

    answer_words = [idx_to_word[idx] for idx in generated_answer if idx in idx_to_word]
    return " ".join(answer_words)


# Main script
if __name__ == "__main__":
    # Paths to files
    vocab_path = "vocab.pkl"
    model_path = "pointer_generator_epoch_3.pth"

    # Load vocabulary and model
    vocab = load_vocab(vocab_path)

    # Ensure special tokens are in the vocabulary
    special_tokens = ["<PAD>", "<START>", "<UNK>"]
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)  # Add the token with a new unique index

    model = PointerGeneratorNetwork(
        len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT
    ).to(device)

    # Load model state
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("encoder.lstm"):
            new_key = key.replace("encoder.lstm", "encoder")
        elif key.startswith("attention.attn"):
            new_key = key.replace("attention.attn", "attention.0")
        else:
            new_key = key
        new_state_dict[new_key] = value

    if "embedding.weight" in new_state_dict:
        saved_embedding = new_state_dict["embedding.weight"]
        current_embedding = model.embedding.weight

        if saved_embedding.size(0) > current_embedding.size(0):
            saved_embedding = saved_embedding[: current_embedding.size(0)]
        elif saved_embedding.size(0) < current_embedding.size(0):
            padding = torch.randn(
                current_embedding.size(0) - saved_embedding.size(0),
                saved_embedding.size(1),
                device=device,
            )
            saved_embedding = torch.cat([saved_embedding, padding], dim=0)

        new_state_dict["embedding.weight"] = saved_embedding

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Provide your custom context and question
    context_text = "The interest rate is 10%"
    question_text = "What is the interest rate?"

    # Generate and print the answer
    answer = generate_answer(model, context_text, question_text, vocab)
    print(f"Context: {context_text}")
    print(f"Question: {question_text}")
    print(f"Answer: {answer}")
