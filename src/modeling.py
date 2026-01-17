import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
import joblib
from typing import Tuple, Any

class BaselineModel:
    def __init__(self, strategy='most_frequent'):
        self.model = DummyClassifier(strategy=strategy, random_state=42)
        self.name = "Baseline (Majority Class)"
    
    def fit(self, X, y):
        """Fitting the baseline model to the training data."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict class labels for the input samples."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for the input samples."""
        return self.model.predict_proba(X)


def create_count_vectorizer(max_features: int = 5000, 
                           ngram_range: Tuple[int, int] = (1, 1),
                           **kwargs) -> CountVectorizer:
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        **kwargs
    )
    return vectorizer


def create_tfidf_vectorizer(max_features: int = 5000,
                           ngram_range: Tuple[int, int] = (1, 1),
                           **kwargs) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        **kwargs
    )
    return vectorizer


def vectorize_text(train_texts, test_texts, vectorizer_type='tfidf', **vectorizer_kwargs):
    # Create vectorizer
    if vectorizer_type == 'count':
        vectorizer = create_count_vectorizer(**vectorizer_kwargs)
    elif vectorizer_type == 'tfidf':
        vectorizer = create_tfidf_vectorizer(**vectorizer_kwargs)
    else:
        raise ValueError("vectorizer_type must be 'count' or 'tfidf'")
    
    # Fit on training data and transform both
    X_train_vec = vectorizer.fit_transform(train_texts)
    X_test_vec = vectorizer.transform(test_texts)
    
    return X_train_vec, X_test_vec, vectorizer


def train_baseline_model(X_train, y_train) -> BaselineModel:
    model = BaselineModel()
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, 
                              max_iter: int = 1000,
                              random_state: int = 42,
                              **kwargs) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train,
                       max_depth: int = 10,
                       random_state: int = 42,
                       **kwargs) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, filepath: str):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def get_feature_importance(model, vectorizer, top_n: int = 20) -> pd.DataFrame:
    feature_names = vectorizer.get_feature_names_out()
    
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        raise ValueError("Model doesn't have coef_ or feature_importances_ attribute")
    
    # Gathering top N features
    top_indices = np.argsort(importance)[-top_n:][::-1]
    
    top_features = pd.DataFrame({
        'feature': [feature_names[i] for i in top_indices],
        'importance': [importance[i] for i in top_indices]
    })
    
    return top_features


def predict_message(message: str, model, vectorizer, clean_func=None):
    if clean_func:
        message = clean_func(message)
    
    # Vectorization
    message_vec = vectorizer.transform([message])
    
    # Predicting
    prediction = model.predict(message_vec)[0]
    probability = model.predict_proba(message_vec)[0]
    
    label = 'spam' if prediction == 1 else 'ham'
    confidence = probability[np.where(model.classes_ == prediction)[0][0]]
    
    return label, confidence