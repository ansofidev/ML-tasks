# Mapping for class names because in the Animals10 dataset the class names are in Italian
CLASS_MAP = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}


def get_english_classes(dataset_classes):
    # Create array with English names for classes
    return [CLASS_MAP[c] for c in dataset_classes]


def idx_to_english(dataset_classes):
    # Create dictionary
    return {i: CLASS_MAP[c] for i, c in enumerate(dataset_classes)}