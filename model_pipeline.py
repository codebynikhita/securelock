import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, 'accounts.csv')

FEATURE_COLS = [
    'network_following_ratio',
    'username_digit_ratio',
    'username_has_trailing_digits',
    'profile_completeness',
    'is_new_and_aggressive',
    'account_age',
    'profile_picture',
    'post_frequency',
    'content_similarity'
]

def engineer_features(df):
    df = df.copy()
    age_days = df['account_age'].copy()
    df['account_age'] = (age_days / 365.0).clip(lower=0.001)
    df['network_following_ratio'] = df['network_count'] / (df['following_count'] + 1)
    df['username_digit_ratio'] = df['username'].apply(
        lambda u: sum(c.isdigit() for c in str(u)) / len(str(u)) if len(str(u)) > 0 else 0
    )
    df['username_has_trailing_digits'] = df['username'].apply(
        lambda u: 1 if re.search(r'\d{4,}$', str(u)) else 0
    )
    bio_present = df['bio'].fillna('').apply(lambda b: 1 if len(str(b).strip()) > 0 else 0)
    df['profile_completeness'] = df['profile_picture'] + bio_present + (df['posts_count'] > 0).astype(int)
    df['is_new_and_aggressive'] = (
        (age_days < 30) &
        ((df['posts_count'] > 300) | (df['following_count'] > 1000))
    ).astype(int)
    df['post_frequency'] = df['posts_count'] / (age_days + 1)
    if 'content_similarity' not in df.columns:
        df['content_similarity'] = 0.1
    else:
        df['content_similarity'] = df['content_similarity'].fillna(0.1)
    return df

def train_realistic_models():
    print("=== SecureLock Model Pipeline Training ===")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset accounts.csv not found at {CSV_PATH}")

    df_raw = pd.read_csv(CSV_PATH)
    df_raw.dropna(subset=['username', 'account_age', 'profile_picture'], inplace=True)

    rename_map = {
        'tweets_count': 'posts_count',
        'followers_count': 'network_count',
        'duplicate_tweets': 'duplicate_posts',
        'name': 'display_name'
    }
    df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns}, inplace=True)
    df_raw['profile_picture'] = df_raw['profile_picture'].apply(lambda x: 1 if x >= 1 else 0)

    df_feat = engineer_features(df_raw)

    y_fake  = df_feat['is_fake'].astype(int)
    y_clone = df_feat['is_clone'].astype(int)
    X = df_feat[FEATURE_COLS]

    X_train, X_test, y_train_fake, y_test_fake = train_test_split(X, y_fake,  test_size=0.2, random_state=42)
    _,       _,      y_train_clone, y_test_clone = train_test_split(X, y_clone, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    print("Scaler saved.")

    # Lightweight RandomForest — fits comfortably in 512MB RAM
    rf_fake = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
    rf_fake.fit(X_train_scaled, y_train_fake)
    joblib.dump(rf_fake, os.path.join(MODEL_DIR, 'ensemble_fake.joblib'))
    print(f"Fake model accuracy: {accuracy_score(y_test_fake, rf_fake.predict(X_test_scaled))*100:.1f}%")

    rf_clone = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
    rf_clone.fit(X_train_scaled, y_train_clone)
    joblib.dump(rf_clone, os.path.join(MODEL_DIR, 'ensemble_clone.joblib'))
    print(f"Clone model accuracy: {accuracy_score(y_test_clone, rf_clone.predict(X_test_scaled))*100:.1f}%")

    knn_fake = KNeighborsClassifier(n_neighbors=7)
    knn_fake.fit(X_train_scaled, y_train_fake)
    joblib.dump(knn_fake, os.path.join(MODEL_DIR, 'knn_fake.joblib'))

    knn_clone = KNeighborsClassifier(n_neighbors=7)
    knn_clone.fit(X_train_scaled, y_train_clone)
    joblib.dump(knn_clone, os.path.join(MODEL_DIR, 'knn_clone.joblib'))

    importances = {FEATURE_COLS[i]: float(rf_fake.feature_importances_[i]) for i in range(len(FEATURE_COLS))}
    joblib.dump(importances, os.path.join(MODEL_DIR, 'feature_importances.joblib'))

    print("All models saved successfully.")

if __name__ == '__main__':
    train_realistic_models()
