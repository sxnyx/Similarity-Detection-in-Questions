# 🧠 Bag of Words with Preprocessing and Advanced Features

This project demonstrates a complete text preprocessing and feature extraction pipeline using the **Bag of Words (BoW)** model. It includes advanced text cleaning, normalization, and tokenization techniques designed to prepare raw text data for machine learning models.

DataSet Link : https://www.kaggle.com/c/quora-question-pairs

---

## 📌 Overview

This notebook walks through the process of:

- **Loading and sampling** a large text dataset for processing.
- **Preprocessing text** using robust methods including:
  - Lowercasing
  - Symbol and number replacement (e.g., ₹ → rupee, 1000000 → 1m)
  - Decontracting (e.g., *can't* → *can not*)
  - HTML tag removal
  - Regular expression-based cleaning
- **Feature extraction** using:
  - Bag of Words (CountVectorizer)
  - TF-IDF (TfidfVectorizer)
  - Unigrams and bigrams
- **Data visualization** to understand class distribution and token frequencies.

---

## 🔧 Libraries Used

| Library              | Description                          |
|----------------------|--------------------------------------|
| `pandas`, `numpy`    | Data manipulation                    |
| `matplotlib`, `seaborn` | Visualization tools               |
| `BeautifulSoup`      | HTML parsing and tag removal         |
| `re` (Regex)         | Text pattern cleaning                |
| `sklearn.feature_extraction.text` | Feature vectorization  |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bow-text-processing.git
   cd bow-text-processing
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook bow-with-preprocessing-and-advanced-features.ipynb
   ```

---

##  Preprocessing Highlights

-The preprocessing pipeline includes:

-Lowercasing text
-Removing special characters, digits, HTML tags
-Replacing common currency/number symbols (e.g., $ → dollar, ₹ → rupee)
-Decontracting (e.g., "won't" → "will not")
-Tokenization and stemming (if needed)
-Removing stopwords (optional)

---

## Feature Extraction

Two main methods are used:
-CountVectorizer (BoW): Represents text as a frequency count of words.
-TfidfVectorizer: Applies Term Frequency-Inverse Document Frequency weighting to highlight important terms.

Both unigram and bigram tokenizations are explored.

---

## 📊 Visualizations

Add charts to show:

- Distribution of classes/labels
- Top 20 most frequent unigrams and bigrams
- Comparison of CountVectorizer vs TfidfVectorizer outputs

<img width="954" height="795" alt="image" src="https://github.com/user-attachments/assets/a0a72145-c26d-49ee-b40d-3c61935f0032" />

---

## 🧠 Next Steps

- Add classification models (e.g., Logistic Regression, Naive Bayes)
- Evaluate using metrics like F1-score, ROC-AUC
- Extend with Word2Vec or transformer-based embeddings (e.g., BERT)

---

## 🤝 Contributing

1. Fork the repo
2. Create a new branch: `feature/your-feature-name`
3. Submit a pull request
