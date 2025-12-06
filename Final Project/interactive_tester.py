#!/usr/bin/env python3
"""
Interactive Emotion Classifier Tester
Allows testing the trained emotion model with custom inputs
"""

import pickle
import sys

def load_model():
    """Load the trained model and vectorizer"""
    try:
        model = pickle.load(open("emotion_model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run 'python train_model.py' first to train the model.")
        sys.exit(1)

def predict_emotion(text, model, vectorizer):
    """Predict emotion for given text"""
    text_vec = vectorizer.transform([text])
    emotion = model.predict(text_vec)[0]
    
    # Get confidence scores
    probabilities = model.predict_proba(text_vec)[0]
    classes = model.classes_
    
    return emotion, dict(zip(classes, probabilities))

def display_header():
    """Display welcome header"""
    print("\n" + "="*70)
    print("🤖 EMOTION CLASSIFIER - INTERACTIVE TESTER")
    print("="*70)
    print("Type your text and press Enter to predict the emotion.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'examples' to see sample texts.")
    print("Type 'help' for more information.")
    print("="*70 + "\n")

def display_examples():
    """Display example sentences"""
    examples = [
        ("I just won the lottery! I'm so excited!", "JOY"),
        ("I haven't felt this terrible in years", "SADNESS"),
        ("This is absolutely infuriating!", "ANGER"),
        ("I'm worried something bad might happen", "FEAR"),
        ("The weather is nice today", "NEUTRAL")
    ]
    
    print("\n📚 EXAMPLE SENTENCES:\n")
    for text, emotion in examples:
        print(f"  • \"{text}\"")
        print(f"    → Expected emotion: {emotion}\n")

def display_help():
    """Display help information"""
    print("\n" + "="*70)
    print("ℹ️  HELP & INFORMATION")
    print("="*70)
    print("""
COMMANDS:
  • Just type any sentence to predict its emotion
  • 'examples'  - Show example test sentences
  • 'help'      - Display this help message
  • 'quit/exit' - Exit the program

EMOTION CATEGORIES:
  🟢 JOY       - Happiness, excitement, positivity
  🔵 SADNESS   - Sorrow, disappointment, melancholy
  🔴 ANGER     - Frustration, irritation, rage
  🟡 FEAR      - Anxiety, nervousness, worry
  ⚪ NEUTRAL   - Objective statements, no clear emotion

TIPS:
  • Use natural, conversational language
  • Longer sentences typically give better results
  • The model learns from the training dataset patterns
  • Confidence scores indicate prediction reliability

MODEL INFORMATION:
  • Algorithm: Logistic Regression
  • Features: TF-IDF vectorization
  • Classes: 5 emotion categories
  • Training data: 2000+ sentences
    
""")
    print("="*70 + "\n")

def format_output(text, emotion, probabilities):
    """Format and display prediction results"""
    emotion_emoji = {
        "joy": "🟢",
        "sadness": "🔵",
        "anger": "🔴",
        "fear": "🟡",
        "neutral": "⚪"
    }
    
    emoji = emotion_emoji.get(emotion.lower(), "❓")
    
    print(f"\n📝 Input: \"{text}\"")
    print(f"\n🎯 PREDICTION:")
    print(f"   {emoji} Emotion: {emotion.upper()}")
    
    print(f"\n📊 CONFIDENCE SCORES:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for emotion_class, prob in sorted_probs:
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"   {emotion_class.upper():10} │{bar}│ {prob*100:5.1f}%")
    
    print()

def main():
    """Main interactive loop"""
    print("\n🔄 Loading model and vectorizer...")
    model, vectorizer = load_model()
    print("✓ Model loaded successfully!\n")
    
    display_header()
    
    while True:
        try:
            user_input = input("Enter text (or type 'help'): ").strip()
            
            if not user_input:
                print("⚠️  Please enter some text.\n")
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Thank you for using the Emotion Classifier!")
                print("="*70 + "\n")
                break
            
            elif user_input.lower() == 'help':
                display_help()
                continue
            
            elif user_input.lower() == 'examples':
                display_examples()
                continue
            
            # Predict emotion
            emotion, probabilities = predict_emotion(user_input, model, vectorizer)
            format_output(user_input, emotion, probabilities)
        
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"❌ Error: {e}\n")
            continue

if __name__ == "__main__":
    main()
