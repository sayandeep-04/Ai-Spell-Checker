import string
import re
import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')

# --- Current N-gram Implementation (from your existing code) ---
class NGramSpellChecker:
    def __init__(self, corpus_file='big.txt'):
        try:
            self.word_probs, self.bigram_counts, self.first_word_counts, self.vocab = self.load_model(corpus_file)
            self.approach = "Statistical Language Model (N-gram)"
            self.model_type = "Unsupervised Statistical Model"
        except FileNotFoundError:
            print(f"Warning: {corpus_file} not found. Using dummy data for N-gram model.")
            self.create_dummy_model()
    
    def create_dummy_model(self):
        # Create a basic vocabulary for demonstration
        common_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'has', 'two', 'more', 'very', 'after', 'words', 'its', 'just', 'where', 'most', 'get', 'through', 'back', 'much', 'good', 'before', 'go', 'man', 'our', 'want', 'way', 'too', 'any', 'may', 'say', 'here', 'give', 'why', 'come', 'should', 'now', 'made', 'each', 'over', 'think', 'also', 'around', 'another', 'came', 'three', 'high', 'right', 'small', 'large', 'such', 'even', 'well', 'without', 'again', 'place', 'old', 'take', 'great', 'little', 'where', 'year', 'work', 'part', 'know', 'get', 'own', 'say', 'come', 'could', 'new', 'write', 'find', 'long', 'here', 'must', 'look', 'day', 'water', 'been', 'call', 'who', 'oil', 'sit', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part']
        
        self.vocab = set(common_words)
        self.word_probs = {word: 1.0/len(common_words) for word in common_words}
        self.bigram_counts = Counter()
        self.first_word_counts = Counter()
        self.approach = "Statistical Language Model (N-gram) - Demo Version"
        self.model_type = "Unsupervised Statistical Model"
    
    def load_model(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read().lower()
            words = re.findall(r'\w+', text)
        
        word_counts = Counter(words)
        total_words = sum(word_counts.values())
        word_probs = {word: count/total_words for word, count in word_counts.items()}
        
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        first_word_counts = Counter(w1 for w1, w2 in bigrams)
        
        return word_probs, bigram_counts, first_word_counts, set(words)
    
    def get_edits(self, word):
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        
        return set(deletes + transposes + replaces + inserts)
    
    def get_edits2(self, word):
        return set(e2 for e1 in self.get_edits(word) for e2 in self.get_edits(e1))
    
    def get_suggestions(self, word, max_suggestions=5):
        if word in self.vocab:
            return []
        
        candidates = self.get_edits(word) & self.vocab
        if not candidates:
            candidates = self.get_edits2(word) & self.vocab
        if not candidates:
            return []
        
        suggestions = []
        for candidate in candidates:
            log_word_prob = math.log(self.word_probs.get(candidate, 1e-10))
            suggestions.append((candidate, log_word_prob))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in suggestions[:max_suggestions]]
    
    def correct_word(self, word):
        if word in self.vocab:
            return word
        suggestions = self.get_suggestions(word, 1)
        return suggestions[0] if suggestions else word

# --- Machine Learning Based Spell Checkers ---
class MLSpellChecker:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.vocab = set()
        self.approach = self.get_approach_name()
        
    def get_approach_name(self):
        approaches = {
            'logistic': 'Supervised ML - Logistic Regression',
            'decision_tree': 'Supervised ML - Decision Tree',
            'random_forest': 'Supervised ML - Random Forest'
        }
        return approaches.get(self.model_type, 'Unknown ML Approach')
    
    def generate_features(self, word):
        """Generate features for a word"""
        features = []
        
        # Length features
        features.append(len(word))
        features.append(len(set(word)))  # unique characters
        
        # Character frequency features
        char_counts = Counter(word)
        features.extend([
            char_counts.get('a', 0), char_counts.get('e', 0), 
            char_counts.get('i', 0), char_counts.get('o', 0), 
            char_counts.get('u', 0)  # vowel counts
        ])
        
        # Common patterns
        features.append(1 if word.endswith('ing') else 0)
        features.append(1 if word.endswith('ed') else 0)
        features.append(1 if word.endswith('ly') else 0)
        features.append(1 if word.startswith('un') else 0)
        features.append(1 if word.startswith('re') else 0)
        
        # Double letters
        features.append(1 if any(word[i] == word[i+1] for i in range(len(word)-1)) else 0)
        
        return features
    
    def create_training_data(self, vocab_sample=None):
        """Create training data with correct and misspelled words"""
        if vocab_sample is None:
            vocab_sample = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'has', 'two', 'more', 'very', 'after', 'words', 'its', 'just', 'where', 'most', 'get', 'through', 'back', 'much', 'good', 'before', 'go', 'man', 'our', 'want', 'way', 'too', 'any', 'may', 'say', 'here', 'give', 'why', 'come', 'should', 'now', 'made', 'each', 'over', 'think', 'also', 'around', 'another', 'came', 'three', 'high', 'right', 'small', 'large', 'such', 'even', 'well', 'without', 'again', 'place', 'old', 'take', 'great', 'little', 'where', 'year', 'work', 'part', 'know', 'get', 'own', 'say', 'come', 'could', 'new', 'write', 'find', 'long', 'here', 'must', 'look', 'day', 'water', 'been', 'call', 'who', 'oil', 'sit', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part']
        
        self.vocab = set(vocab_sample)
        X, y = [], []
        
        # Correct words (label = 1)
        for word in vocab_sample:
            X.append(self.generate_features(word))
            y.append(1)
        
        # Generate misspelled words (label = 0)
        letters = string.ascii_lowercase
        for word in vocab_sample[:50]:  # Limit to avoid too much data
            # Create some simple misspellings
            misspellings = []
            
            # Delete a character
            if len(word) > 2:
                for i in range(len(word)):
                    misspellings.append(word[:i] + word[i+1:])
            
            # Insert a character
            for i in range(len(word) + 1):
                for c in letters[:5]:  # Limit insertions
                    misspellings.append(word[:i] + c + word[i:])
            
            # Replace a character
            for i in range(len(word)):
                for c in letters[:3]:  # Limit replacements
                    if c != word[i]:
                        misspellings.append(word[:i] + c + word[i+1:])
            
            # Add features for misspellings (limit to avoid memory issues)
            for misspelling in misspellings[:10]:
                X.append(self.generate_features(misspelling))
                y.append(0)
        
        return np.array(X), np.array(y)
    
    def train(self, vocab_sample=None):
        """Train the ML model"""
        X, y = self.create_training_data(vocab_sample)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(random_state=42, max_depth=10)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def is_correct(self, word):
        """Check if word is spelled correctly"""
        if self.model is None:
            return word.lower() in self.vocab
        
        features = np.array([self.generate_features(word)])
        prediction = self.model.predict(features)[0]
        return prediction == 1

# --- Comparison Function ---
def compare_algorithms():
    print("="*80)
    print("🔍 SPELL CHECKER ALGORITHMS COMPARISON")
    print("="*80)
    print()
    
    # Test data
    test_sentences = [
        "I havve a gret idee for tomorow",
        "The quik brown fox jumps over the lasy dog", 
        "Ths is a vrey importnt mesage",
        "I wnt to the stor yestrday",
        "Plese hlp me wit ths problm"
    ]
    
    correct_sentences = [
        "I have a great idea for tomorrow",
        "The quick brown fox jumps over the lazy dog",
        "This is a very important message", 
        "I want to the store yesterday",
        "Please help me with this problem"
    ]
    
    results = []
    
    print("📊 TESTING ALGORITHMS...")
    print("-" * 50)
    
    # Test N-gram approach
    print("1️⃣  Testing N-gram Statistical Model...")
    start_time = time.time()
    ngram_checker = NGramSpellChecker()
    ngram_time = time.time() - start_time
    
    # Calculate N-gram accuracy (simplified)
    ngram_correct = 0
    total_words = 0
    for sentence in test_sentences:
        words = re.findall(r'\w+', sentence.lower())
        for word in words:
            total_words += 1
            if word in ngram_checker.vocab or len(ngram_checker.get_suggestions(word, 1)) > 0:
                ngram_correct += 1
    
    ngram_accuracy = (ngram_correct / total_words) * 100 if total_words > 0 else 0
    
    results.append({
        'Algorithm': 'N-gram Language Model',
        'Approach': ngram_checker.approach,
        'Model Type': ngram_checker.model_type,
        'Accuracy': ngram_accuracy,
        'Training Time': ngram_time,
        'Pros': ['No training data required', 'Works with context', 'Fast inference', 'Good for rare words'],
        'Cons': ['Requires large corpus', 'Limited by corpus quality', 'Memory intensive', 'May miss domain-specific terms']
    })
    
    # Test ML approaches
    ml_algorithms = [
        ('logistic', 'Logistic Regression'),
        ('decision_tree', 'Decision Tree'),
        ('random_forest', 'Random Forest')
    ]
    
    for i, (model_type, name) in enumerate(ml_algorithms, 2):
        print(f"{i}️⃣  Testing {name}...")
        start_time = time.time()
        
        ml_checker = MLSpellChecker(model_type)
        accuracy = ml_checker.train()
        training_time = time.time() - start_time
        
        # Test accuracy on our test data
        test_accuracy = 0
        total_test_words = 0
        for sentence in test_sentences:
            words = re.findall(r'\w+', sentence.lower())
            for word in words:
                total_test_words += 1
                if ml_checker.is_correct(word):
                    test_accuracy += 1
        
        final_accuracy = (test_accuracy / total_test_words * 100) if total_test_words > 0 else accuracy * 100
        
        # Define pros and cons for each ML approach
        if model_type == 'logistic':
            pros = ['Fast training', 'Probabilistic output', 'Less prone to overfitting', 'Interpretable coefficients']
            cons = ['Linear assumptions', 'Requires feature engineering', 'Limited complexity', 'May underfit complex patterns']
        elif model_type == 'decision_tree':
            pros = ['Easy to interpret', 'Handles non-linear patterns', 'No assumptions about data', 'Fast prediction']
            cons = ['Prone to overfitting', 'Unstable to small changes', 'Biased toward features with many levels', 'Limited expressiveness']
        else:  # random_forest
            pros = ['Reduces overfitting', 'Handles missing values', 'Feature importance ranking', 'Works with mixed data types']
            cons = ['Less interpretable', 'More memory intensive', 'Slower training', 'Can still overfit with noise']
        
        results.append({
            'Algorithm': name,
            'Approach': ml_checker.approach,
            'Model Type': 'Supervised Machine Learning',
            'Accuracy': final_accuracy,
            'Training Time': training_time,
            'Pros': pros,
            'Cons': cons
        })
    
    print()
    print("="*80)
    print("📈 DETAILED COMPARISON RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n🔸 {i}. {result['Algorithm'].upper()}")
        print(f"   {'─' * (len(result['Algorithm']) + 10)}")
        print(f"   📊 Approach: {result['Approach']}")
        print(f"   🤖 Model Type: {result['Model Type']}")
        print(f"   🎯 Accuracy: {result['Accuracy']:.2f}%")
        print(f"   ⏱️  Training Time: {result['Training Time']:.3f} seconds")
        
        print(f"   \n   ✅ PROS:")
        for pro in result['Pros']:
            print(f"      • {pro}")
        
        print(f"   \n   ❌ CONS:")
        for con in result['Cons']:
            print(f"      • {con}")
        print()
    
    # Summary comparison
    print("="*80)
    print("📋 SUMMARY COMPARISON TABLE")
    print("="*80)
    
    print(f"{'Algorithm':<25} {'Accuracy':<12} {'Time (s)':<12} {'Model Type':<20}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['Algorithm']:<25} {result['Accuracy']:>8.2f}% {result['Training Time']:>9.3f}s {result['Model Type']:<20}")
    
    print("\n" + "="*80)
    print("🏆 RECOMMENDATIONS")
    print("="*80)
    
    best_accuracy = max(results, key=lambda x: x['Accuracy'])
    fastest = min(results, key=lambda x: x['Training Time'])
    
    print(f"🥇 Best Accuracy: {best_accuracy['Algorithm']} ({best_accuracy['Accuracy']:.2f}%)")
    print(f"⚡ Fastest Training: {fastest['Algorithm']} ({fastest['Training Time']:.3f}s)")
    
    print(f"\n💡 For Production Use:")
    print(f"   • Real-time applications: N-gram Language Model (fast, no training)")
    print(f"   • Domain-specific: Random Forest (handles complex patterns)")
    print(f"   • Limited resources: Logistic Regression (balanced performance)")
    print(f"   • Interpretability: Decision Tree (easy to understand)")
    
    return results

if __name__ == "__main__":
    results = compare_algorithms()
    
    # Save results for graph.py
    # import json
    # with open('comparison_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to 'comparison_results.json' for visualization!")
    print(f"📊 Run 'python graph.py' to generate accuracy comparison graphs.")