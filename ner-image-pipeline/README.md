# Task 2 — NER + Image Classification Pipeline

The goal of this task was to build a pipeline that verifies whether a text statement about an image is correct.

Input:
- text
- image

Output:
- boolean value (True / False)

## Models

Two models are used:

### NER (NLP)

**Model:** fine-tuned **DistilBERT**

**Task:** extract animal names from text.

**Classes:** dog, cat, cow, horse, sheep, elephant, butterfly, chicken, spider, squirrel

**Dataset:** generated using `dataset_generator.py`

**Example:**
"There is a cow in the picture" → ["cow"]

**Evaluation metric:**
- F1-score ≈ 1

---

### Image Classification

**Model:** **ResNet18**

**Classes:** dog, cat, cow, horse, sheep, elephant, butterfly, chicken, spider, squirrel

**Dataset:** Animals10 (10 animal classes)

**Evaluation metric:**
- Accuracy ≈ 0.85 (trained for 5 epochs)

---

## Pipeline

Steps:

1. Extract animal names from the text using the NER model.
2. Predict the animal class in the image.
3. Compare predictions.
4. Return `True` if the statement matches the image, otherwise `False`.

## Demo

The repository includes a Jupyter Notebook (`demo.ipynb`) showing:

- data loading
- training process
- evaluation
- pipeline execution
- examples
- edge cases

## Run

1. `pip install -r requirements.txt`
2. Open `demo.ipynb` and run all its cells


