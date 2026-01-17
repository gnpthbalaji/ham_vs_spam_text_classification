import pandas as pd
import numpy as np
import re
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding='latin-1')
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    return df

def missing_value_check(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    return {
        'total_missing': missing.sum(),
        'missing_per_column': missing[missing > 0].to_dict()
    }

def missing_value_handling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=['label'])
    df['message'] = df['message'].fillna('')
    return df

def labels_encoding(df:pd.DataFrame, label_col: str='label') -> pd.DataFrame:
    df = df.copy()
    df['label_encoded'] = df[label_col].map({'ham': 0, 'spam': 1})
    return df

def clean_text(text:str, 
               lowercase: bool=True, 
                remove_punctuation: bool=True,
                remove_extra_whitespace: bool=True) -> str:
    if not isinstance(text, str):
        return ''
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    if remove_extra_whitespace:
        text = ' '.join(text.split())
    return text

def preprocess_data(df: pd.DataFrame, 
                    text_col: str='message', 
                    **clean_kwargs) -> pd.DataFrame:
    df = df.copy()
    df['cleaned_message'] = df[text_col].apply(lambda x: clean_text(x, **clean_kwargs))
    return df

def get_text_stats(df: pd.DataFrame, 
                   text_col: str='message') -> pd.DataFrame:
    df = df.copy()
    df['message_length'] = df[text_col].apply(len)
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    df['avg_word_length'] = df[text_col].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
    return df

def data_split(df: pd.DataFrame,
               test_size: float=0.2,
               random_state: int=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label_encoded'] 
                                         if 'label_encoded' in df.columns else None)
    return train_df, test_df

def get_class_distribution(df: pd.DataFrame, label_col: str='label') -> pd.DataFrame:
    counts = df[label_col].value_counts()
    percentages = df[label_col].value_counts(normalize=True) * 100
    distribution = pd.DataFrame({'count': counts, 'percentage': percentages})
    return distribution

def get_sample_message (df: pd.DataFrame, 
                        label: str,
                        n:int = 5,
                        label_col: str='label',
                        text_col: str='message') -> list: 
    samples = df[df[label_col] == label][text_col].sample(n=min(n, len(df[df[label_col] == label])))
    return samples.tolist()
