# AI Spell Checker

An NLP-based **AI Spell Checker** that detects misspelled words and generates contextually relevant spelling correction suggestions using **Unigram and Bigram N-gram Language Models**.

The project uses statistical language modeling, text preprocessing, tokenization, frequency analysis, and smoothing techniques to improve the quality of spelling suggestions. It also provides an interactive **Streamlit** interface for real-time text input and correction.

## 🚀 Features

* Real-time spelling correction
* Unigram language model
* Bigram language model
* Context-aware correction suggestions
* Text preprocessing and normalization
* Word tokenization
* Word-frequency analysis
* Smoothing for handling unseen words and word combinations
* Interactive Streamlit web interface
* Comparison and evaluation of correction results

## 🛠️ Technologies Used

* **Python**
* **Natural Language Processing (NLP)**
* **N-gram Language Models**
* **Streamlit**
* **Regular Expressions**
* **Statistical Language Modeling**
* **JSON**

## 🧠 How It Works

The system follows a series of NLP processing steps:

```text
Input Text
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Word Frequency Analysis
    ↓
Unigram & Bigram Models
    ↓
Candidate Generation
    ↓
Smoothing
    ↓
Context-Based Ranking
    ↓
Spelling Suggestions
```

### 1. Text Preprocessing

The input corpus is normalized and processed to prepare it for language modeling.

The system performs operations such as:

* Converting text to lowercase
* Tokenization
* Removing unnecessary punctuation
* Preparing words for frequency analysis

### 2. Unigram Model

The unigram model calculates the frequency/probability of individual words.

It helps determine how commonly a candidate word occurs in the corpus.

### 3. Bigram Model

The bigram model considers pairs of consecutive words.

This provides contextual information and helps the system choose a correction that fits better with surrounding words.

### 4. Smoothing

Smoothing is applied to handle words or word combinations that are not present in the training corpus.

This prevents unseen combinations from receiving an unusable probability of zero.

### 5. Candidate Ranking

Potential corrections are evaluated using the statistical language models. Unigram and bigram information are combined to identify more contextually appropriate suggestions.

## 📁 Project Structure

```text
AI-Spell-Checker/
│
├── app.py
├── big.txt
├── comparison.py
├── comparison_results.json
└── README.md
```

### File Description

| File                      | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `app.py`                  | Main Streamlit application and spelling-correction interface |
| `big.txt`                 | Text corpus used for language-model training                 |
| `comparison.py`           | Script used for comparing/evaluating correction approaches   |
| `comparison_results.json` | Stores comparison/evaluation results                         |
| `README.md`               | Project documentation                                        |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sayandeep-04/Ai-Spell-Checker.git
```

### 2. Navigate to the project

```bash
cd Ai-Spell-Checker
```

### 3. Install the required dependencies

Make sure Python is installed on your system.

Install Streamlit:

```bash
pip install streamlit
```

If the project contains additional dependencies, install them according to the project's environment or requirements file.

## ▶️ Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

After starting the application, Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use the application.

## 💡 Example

The user enters a sentence containing a spelling mistake.

```text
Input:
I recieved the mesage yesterday.
```

The system analyzes the words and surrounding context and generates possible corrections such as:

```text
received
message
```

The language model helps rank candidate words according to their frequency and contextual probability.

## 📊 Evaluation

The project includes `comparison.py` and `comparison_results.json` for comparing correction results and evaluating the behavior of the implemented approaches.

This helps analyze how different language-model strategies perform when generating spelling suggestions.

## 🔮 Future Improvements

Possible improvements include:

* Adding a larger and more diverse corpus
* Implementing edit-distance-based candidate generation
* Adding advanced contextual language models
* Supporting multiple languages
* Adding grammar correction
* Improving correction ranking
* Integrating transformer-based NLP models
* Deploying the application online
* Adding an API for external applications

## 👨‍💻 Author

**Sayandeep Roy**

B.Tech — Information Technology
Asansol Engineering College, West Bengal

GitHub:
https://github.com/sayandeep-04

LinkedIn:
https://linkedin.com/in/sayandeep-roy-b8396b2b3

## 📄 License

This project is intended for educational and development purposes.
