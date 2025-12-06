import pickle

# Load model and vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Test with various inputs
test_sentences = [
    "I'm so happy today!",
    "I feel terrible and sad",
    "This makes me so angry!",
    "I'm scared and anxious",
    "That's interesting"
]

print("=" * 60)
print("EMOTION CLASSIFIER TEST")
print("=" * 60)

for sentence in test_sentences:
    text_vec = vectorizer.transform([sentence])
    emotion = model.predict(text_vec)[0]
    print(f"\nText: {sentence}")
    print(f"Emotion: {emotion.upper()}")

print("\n" + "=" * 60)
print("Model test completed successfully!")
print("=" * 60)
