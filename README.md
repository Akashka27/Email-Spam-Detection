# Email Spam Detection System ✉️🛡️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.0+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.12+-red)
![NLTK](https://img.shields.io/badge/NLTK-3.7-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features
- 95% accurate spam detection
- Real-time email classification
- Natural Language Processing (NLP) pipeline
- Interactive probability visualization
- Support for multiple email formats

## 🚀 Quick Start
### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
git clone https://github.com/Akashka27/Email-Spam-Detection.git
cd Email-Spam-Detection
pip install -r requirements.txt
```
### Launch App
```bash
streamlit run app.py
```

## 📊 Dataset

**Combined from various public sources including:**
- Enron-Spam datasets
- SpamAssassin public corpus
- Custom collected samples

📥 [Download Dataset](data/spam.csv)

## 🧠 Model Architecture
```mermaid
graph TD
    A[Raw Email] --> B[Text Preprocessing]
    B --> C[Feature Extraction]
    C --> D[Naive Bayes Classifier]
    D --> E[Spam/Ham Prediction]
```
## 🖼️ System Workflow

- User inputs email text email text
- System processes text (tokenization, stemming, etc.)
- TF-IDF vectorization transforms text to features
- Trained model makes prediction with confidence score
- Results displayed with visual indicators



## 🖼️ Screenshots
![App Screenshot](images/input_screen.png)
![App Screenshot](images/result_screen.pmg.png)

## 🤝 Contributing
Pull requests welcome! Please open an issue first.

## 📜 License
MIT © Akash 
