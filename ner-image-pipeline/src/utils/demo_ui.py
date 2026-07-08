from IPython.display import display
from PIL import Image
from src.utils.labels import CLASS_MAP

def pipeline_demo(pipeline, text, image_path):
    POSSIBLE_CLASSES = list(CLASS_MAP.values())
    print("Possible classes:", ", ".join(POSSIBLE_CLASSES))
    result = pipeline.run(text, image_path)
    img = Image.open(image_path)
    img.thumbnail((300, 300))
    display(img)
    print("Text:", text)
    print("Animals in text:", result["animals_in_text"])
    print("Image prediction:", result["image_prediction"])
    print("Match:", result["result"])