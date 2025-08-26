from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load the tokenizer and model
model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_classifier = pipeline("text-classification", 
                              model = model_name, 
                              tokenizer = model_name, 
                              top_k = None)

# Uses the model to detect what emotion is in the sentence; outputs emotion and score (0-1) of the emotion
def detect_emotion(text):
    results = emotion_classifier(text)[0]
    sorted_results = sorted(results, key = lambda x: x['score'], reverse = True)
    top = sorted_results[0]
    print(f"Text: {text}")
    print(f"Predicted Emotion: {top['label']} ({top['score']:.2f})\n")

# This model can detect 7 emotions
# Joy
# Sadness
# Anger
# Fear
# Neutral
# Love
# Surprise

# Trying some examples
detect_emotion("I can't stop smiling, today has been wonderful!")
detect_emotion("Why would you do that? I'm really upset.")
detect_emotion("I'm just so tired of everything.")
detect_emotion("That was unexpected! Whoa!")