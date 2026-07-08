import json
import random
import os

# Animal classes (they match the classes from the Animals10 image dataset)
animals = [
    "cat","dog","cow","horse","sheep",
    "elephant","butterfly","chicken","spider","squirrel"
]

positive_templates = [
    ["There","is","a","{animal}","in","the","picture","."],
    ["I","see","a","{animal}","."],
    ["This","looks","like","a","{animal}","."],
    ["The","image","contains","a","{animal}","."],
    ["I","think","it","is","a","{animal}","."],
    ["Looks","like","a","{animal}","to","me","."],
    ["Probably","a","{animal}","."],
    ["The","animal","might","be","a","{animal}","."],
    ["It","seems","to","be","a","{animal}","."],
    ["Maybe","it","is","a","{animal}","."],
]

# Negative examples help the model learn when there is NO entity
negative_templates = [
    ["I","cannot","identify","the","animal","."],
    ["The","image","is","unclear","."],
    ["I","do","not","see","any","animal","."],
    ["The","picture","is","too","blurry","."],
    ["There","might","not","be","an","animal","here","."],
    ["This","does","not","look","like","an","animal","."],
]

# Data labels for NER training
label2id = {"O":0,"B-ANIMAL":1,"I-ANIMAL":2}

data = []

for animal in animals:
    for template in positive_templates:
        tokens = []

        # Replace {animal} placeholder with the animal name
        for t in template:
            if t == "{animal}":
                tokens.append(animal)
            else:
                tokens.append(t)

        labels = []

        # Assign B-ANIMAL labels for animal and everything else → O
        for token in tokens:
            if token == animal:
                labels.append(label2id["B-ANIMAL"])
            else:
                labels.append(label2id["O"])

        data.append({
            "tokens": tokens,
            "ner_tags": labels,
            "label": animal,
            "has_animal": True 
        })


for template in negative_templates:
    tokens = template

    # No animals mentioned, so all labels are "O".
    labels = [label2id["O"] for _ in tokens]

    data.append({
        "tokens": tokens,
        "ner_tags": labels,
        "label": None,
        "has_animal": False
    })


# Adds small variations to make sentences less repetitive
augmented = []

for item in data:
    tokens = item["tokens"][:]
    labels = item["ner_tags"][:]

    if random.random() < 0.3 and len(tokens) > 4:
        tokens.insert(0, "Well")
        labels.insert(0, label2id["O"])

    if random.random() < 0.3:
        tokens.append("...")
        labels.append(label2id["O"])

    augmented.append({
        "tokens": tokens,
        "ner_tags": labels,
        "label": item["label"],
        "has_animal": item["has_animal"]
    })

# Combine original and augmented examples
data.extend(augmented)
# Shuffle to avoid ordered patterns
random.shuffle(data)


# Data splitting
split = int(len(data)*0.8)
train_data = data[:split]
val_data = data[split:]


# Save datasets to corresponding files
os.makedirs("data/ner",exist_ok=True)

with open("data/ner/train.json","w") as f:
    json.dump(train_data,f,indent=2)

with open("data/ner/validation.json","w") as f:
    json.dump(val_data,f,indent=2)
