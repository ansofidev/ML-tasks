from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score

LABELS = ["O", "ANIMAL"]
# Create dictionaries for labels
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

def tokenize_and_align_labels(examples, tokenizer):
    # Text tokenization
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,  # Truncate text if it is too long
        padding="max_length",  # Pad sentences to the maximum length so all batches have the same size
        is_split_into_words=True,  # Indicates that the input text is already split into words
    )

    # Array with normalized labels
    aligned_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        # Assign each token to its corresponding word
        word_ids = tokenized.word_ids(batch_index=i)
        # Array with labels for tokens
        label_ids = []

        for word_idx in word_ids:
            # If token does not correspond to a real word
            if word_idx is None:
                # -100 tells the model to ignore this token during training
                label_ids.append(-100)
            else:
                # Assign the label of the corresponding word
                label_ids.append(labels[word_idx])

        # Add processed batch to the result list
        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels

    return tokenized


def compute_metrics(p):
    predictions, labels = p
    # Select the most probable class
    preds = predictions.argmax(-1)

    true_preds = []
    true_labels = []

    for pred, lab in zip(preds, labels):
        curr_preds = []
        curr_labels = []

        for p_i, l_i in zip(pred, lab):
            # Skip special tokens
            if l_i != -100:
                curr_preds.append(ID2LABEL[p_i])
                curr_labels.append(ID2LABEL[l_i])

        true_preds.append(curr_preds)
        true_labels.append(curr_labels)

    # Calculate F1-score
    return {"f1": f1_score(true_labels, true_preds)}


def train_ner(train_dataset, val_dataset):
    # Load tokenizer for DistilBERT
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True
    )

    val_dataset = val_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True
    )
    
    # Load pretrained model
    model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-cased",
        num_labels=len(LABELS),  # Number of classes
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training configuration
    args = TrainingArguments(
        output_dir="models/ner_model",
        learning_rate=3e-5,  # Controls how much the model weights change during training
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,  # Number of times the model will iterate over the entire training dataset
        weight_decay=0.01,  # Regularization parameter to prevent overfitting
        logging_steps=10,
        save_strategy="epoch",  # Save model checkpoint after every epoch
        disable_tqdm=False # Show progress bar
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer), # Responsible for forming correct batches
        compute_metrics=compute_metrics
    )

    # Model training
    trainer.train()

    # Save model and tokenizer
    trainer.save_model("models/ner_model")
    tokenizer.save_pretrained("models/ner_model")

    return trainer