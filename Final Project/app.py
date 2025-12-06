from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load pre-trained model and vectorizer
try:
    model = pickle.load(open("emotion_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("✓ Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please run 'python train_model.py' first to train the model.")
    model = None
    vectorizer = None

# Emotion-specific responses with empathy and support
responses = {
    "joy": "That's wonderful! I'm happy for you 😊",
    "sadness": "I'm sorry you're feeling sad. I'm here for you 💙",
    "anger": "I understand you're upset — tell me what happened 😤",
    "fear": "It's okay to feel scared. You're not alone 🤍",
    "neutral": "Thanks for sharing. Tell me more 🙂"
}

@app.route("/")
def index():
    """Serve the main chatbot interface"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict emotion from user text and return response"""
    try:
        if model is None or vectorizer is None:
            return jsonify({
                "error": "Model not loaded. Please restart the application."
            }), 500
        
        # Get user input
        user_text = request.form.get("text", "").strip()
        
        if not user_text:
            return jsonify({"error": "Please enter some text"}), 400
        
        # Vectorize and predict
        text_vec = vectorizer.transform([user_text])
        emotion = model.predict(text_vec)[0]
        
        # Get appropriate response
        reply = responses.get(emotion, "I'm here for you ❤️")
        
        return jsonify({
            "emotion": emotion.lower(),
            "reply": reply
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({
            "error": "An error occurred while processing your text."
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    model_loaded = model is not None and vectorizer is not None
    return jsonify({
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_ready": model_loaded
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 Emotion-Aware Chatbot - Starting Flask Server")
    print("="*60)
    print("Opening http://localhost:5000 in your browser...")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host="localhost", port=5000)
