import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification


MODEL_PATH = "models/ner_model"
LABELS = ["O", "ANIMAL"]

# Load the trained model and tokenizer
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForTokenClassification.from_pretrained(MODEL_PATH)

    model.eval()  # Enable evaluation mode to disable training mechanisms

    return model, tokenizer


def predict(model, tokenizer, text):
    # Split the sentence into words
    tokens = text.split()

    # Tokenize input
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,  # Indicates that the input text is already split into words
        return_tensors="pt"  # Return results as PyTorch tensors
    )

    # Disable gradient calculation because we only need inference (evaluation)
    with torch.no_grad():
        outputs = model(**inputs)  # Returns prediction scores for each token


    # Select the most probable class for each token
    predictions = torch.argmax(outputs.logits, dim=2)
    # Map tokens back to original words
    word_ids = inputs.word_ids()
    # Result list
    animals = []

    for i, word_id in enumerate(word_ids):
        # Skip tokens that do not correspond to real words
        if word_id is None:
            continue

        # Convert predicted class index to label
        label = LABELS[predictions[0][i]]

        # If the predicted label is ANIMAL, add the word to the result list
        if label == "ANIMAL":
            animals.append(tokens[word_id])

    return animals
