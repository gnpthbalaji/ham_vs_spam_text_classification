# Project: Ham vs. Spam (Text Classification)

### Overview

    * In this project you will build a machine learning model that predicts whether a text message is ham (normal) or spam (unwanted/advertising/scam). This is one of the most common “real world” classification problems and is a great way to practice the full data science workflow.

    * You will work with a labeled dataset of SMS messages and train a model using features created from text (for example: bag-of-words or TF–IDF).

### Learning Goals

By the end of this project, you should be able to:

    1. Load and explore a real dataset with text + labels
    2. Clean and preprocess text (basic normalization)
    3. Convert text to numerical features with CountVectorizer or TfidfVectorizer
    4. Train at least one classification model (baseline + improved)
    5. Evaluate with a confusion matrix, precision, recall, F1, and optionally an ROC curve
    6. Explain tradeoffs (false positives vs. false negatives) in a spam filter

### Dataset

    We will use the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data) ham/spam labeled messages.

        Each row is a text message with a label: ham or spam
        You’ll treat spam as the “positive” class

    What you will submit
        A Colab notebook with your code, outputs, and written interpretation
        Clear evaluation and discussion of results
        A short “recommendation” at the end: what model would you use and why?

### Project Requirements

    1 Data Exploration (EDA)
        At minimum, include:
            * Dataset size (rows, columns)
            * How many spam vs. ham (counts + percentages)
            * Example messages from each class (a few)
            * Any quick text stats you think are useful (optional)
            * message length
            * word counts
            * most common words (with caution about stopwords)
    2 Preprocessing
        At minimum:
            Handle missing values (if any)
            Convert labels to numeric (spam = 1, ham = 0)
            Basic text cleaning (keep it simple)
            lowercase
            optional: remove punctuation
            optional: remove extra whitespace
            Important: Don’t overcomplicate preprocessing. The goal is to learn the workflow.

    3 Modeling
        You must include:
            A baseline model
            Example: predict the majority class (“always ham”)
        At least one real ML model using text features
            Recommended starters:
            Decision Tree
            Logistic Regression
        You must use one of:
            CountVectorizer (bag-of-words)
            TfidfVectorizer (TF–IDF)

    4 Evaluation
        You must report:
            Confusion matrix
            Precision
            Recall
            F1 score
            Accuracy (but do not rely only on accuracy)
        You must include a short interpretation:
            What errors does your model make?
            Which type of mistake is worse for a spam filter: false positives or false negatives?
            Based on your results, would you deploy this model? Why or why not?

    5 Reflection / Communication
        In your final markdown section:
            Summarize your model performance in plain English
            Suggest one realistic improvement you would try next
                hyperparameter tuning
                better text cleaning
                n-grams
                class weighting / threshold tuning
                trying a different model

### Recommended Workflow (Suggested Notebook Structure)

    Section A — Setup
        Import libraries
        Load the dataset
    Section B — Quick EDA
        class balance
        example messages
        message length stats
    Section C — Train/Test Split
        split into train and test
        keep labels aligned
    Section D — Vectorization
        fit vectorizer on training text
        transform training + test text
    Section E — Baseline Model
        majority class predictor
        evaluate
    Section F — ML Model(s)
        train model
        evaluate
        compare to baseline
    Section G — Conclusion
        Which model performed best?
        What tradeoff do you prefer and why?

### Stretch Goals (Optional)

    If you finish early, try one:
        Compare CountVectorizer vs TF–IDF
        Add bigrams (ngram_range=(1,2))
        Try threshold tuning using predicted probabilities
        Plot an ROC curve and discuss what it means
        Inspect the most influential words (model interpretation)
