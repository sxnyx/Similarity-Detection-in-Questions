# Similarity-Detection-in-Questions

DataSet Link : https://www.kaggle.com/c/quora-question-pairs

# 🧠 Bag of Words with Preprocessing and Advanced Features

This project demonstrates a complete text preprocessing and feature extraction pipeline using the **Bag of Words (BoW)** model. It includes advanced text cleaning, normalization, and tokenization techniques designed to prepare raw text data for machine learning models.

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

## 📁 File Structure

```
.
├── bow-with-preprocessing-and-advanced-features.ipynb
├── train.csv
├── README.md
└── (optional) screenshots/
```

---

## ✨ Preprocessing Highlights

The `preprocess()` function performs:

- Normalization of currency symbols and large numbers
- Decontraction (e.g., "isn't" → "is not")
- Lowercasing and trimming
- HTML tag stripping using BeautifulSoup
- Regex-based noise removal

```python
def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ').replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    q = BeautifulSoup(q, 'lxml').get_text()
    q = re.sub(r'[^a-zA-Z]', ' ', q)
    q = re.sub(r'\s+', ' ', q)
    return q
```

---

## 📊 Visualizations

Add charts to show:

- Distribution of classes/labels
- Top 20 most frequent unigrams and bigrams
- Comparison of CountVectorizer vs TfidfVectorizer outputs

> You can add screenshots like:
> 
> ![Example Preprocessing](screenshots/preprocessing.png)

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

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 💬 Feedback

If you have any feedback or suggestions, feel free to open an issue or reach out!
