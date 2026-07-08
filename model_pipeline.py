import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from model_def import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, 'accounts.csv')

# Define aligned 9 feature columns (including content_similarity)
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
    """
    Extracts engineered features matching the exact rules of fake/clone behavior.
    """
    df = df.copy()
    
    # Store age in days for post frequency and aggressive checks
    age_days = df['account_age'].copy()
    
    # Convert age to years
    df['account_age'] = (age_days / 365.0).clip(lower=0.001)
    
    # 1. Follower-to-Following Ratio (followers count is renamed to network_count)
    df['network_following_ratio'] = df['network_count'] / (df['following_count'] + 1)
    
    # 2. Username Patterns
    df['username_digit_ratio'] = df['username'].apply(
        lambda u: sum(c.isdigit() for c in str(u)) / len(str(u)) if len(str(u)) > 0 else 0
    )
    df['username_has_trailing_digits'] = df['username'].apply(
        lambda u: 1 if re.search(r'\d{4,}$', str(u)) else 0
    )
    
    # 3. Profile Completeness
    bio_present = df['bio'].fillna('').apply(lambda b: 1 if len(str(b).strip()) > 0 else 0)
    df['profile_completeness'] = df['profile_picture'] + bio_present + (df['posts_count'] > 0).astype(int)
    
    # 4. Account Age vs Activity (New account < 30 days and aggressive posts or following)
    df['is_new_and_aggressive'] = (
        (age_days < 30) & 
        ((df['posts_count'] > 300) | (df['following_count'] > 1000))
    ).astype(int)
    
    # Post frequency (posts per day)
    df['post_frequency'] = df['posts_count'] / (age_days + 1)
    
    # 5. Content Similarity
    if 'content_similarity' not in df.columns:
        df['content_similarity'] = 0.1
    else:
        df['content_similarity'] = df['content_similarity'].fillna(0.1)
        
    return df

def train_realistic_models():
    print("=== SecureLock Model Pipeline Training ===")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset accounts.csv not found at {CSV_PATH}")
        
    # Load raw dataset
    df_raw = pd.read_csv(CSV_PATH)
    df_raw.dropna(subset=['username', 'account_age', 'profile_picture'], inplace=True)
    
    # Rename columns to match expected schema
    rename_map = {
        'tweets_count': 'posts_count',
        'followers_count': 'network_count',
        'duplicate_tweets': 'duplicate_posts',
        'name': 'display_name'
    }
    df_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_raw.columns}, inplace=True)
    
    # Clean profile picture anomalous values
    df_raw['profile_picture'] = df_raw['profile_picture'].apply(lambda x: 1 if x >= 1 else 0)
    
    # Run feature engineering
    df_feat = engineer_features(df_raw)
    
    # Labels
    y_fake = df_feat['is_fake'].astype(int)
    y_clone = df_feat['is_clone'].astype(int)
    
    X = df_feat[FEATURE_COLS]
    
    # Split
    X_train, X_test, y_train_fake, y_test_fake = train_test_split(X, y_fake, test_size=0.2, random_state=42)
    _, _, y_train_clone, y_test_clone = train_test_split(X, y_clone, test_size=0.2, random_state=42)
    
    # Fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    
    # Build Ensemble models
    # Fake Detector Ensemble
    rf_fake = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    xgb_fake = XGBClassifier(eval_metric='logloss', objective='binary:logistic', n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42)
    ensemble_fake = VotingClassifier(estimators=[('rf', rf_fake), ('xgb', xgb_fake)], voting='soft')
    ensemble_fake.fit(X_train_scaled, y_train_fake)
    
    # Clone Detector Ensemble
    rf_clone = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    xgb_clone = XGBClassifier(eval_metric='logloss', objective='binary:logistic', n_estimators=200, max_depth=6, learning_rate=0.08, random_state=42)
    ensemble_clone = VotingClassifier(estimators=[('rf', rf_clone), ('xgb', xgb_clone)], voting='soft')
    ensemble_clone.fit(X_train_scaled, y_train_clone)
    
    # Train KNN models for anomaly boundaries
    knn_fake = KNeighborsClassifier(n_neighbors=7)
    knn_fake.fit(X_train_scaled, y_train_fake)
    
    knn_clone = KNeighborsClassifier(n_neighbors=7)
    knn_clone.fit(X_train_scaled, y_train_clone)
    
    # Save all models
    joblib.dump(ensemble_fake, os.path.join(MODEL_DIR, 'ensemble_fake.joblib'))
    joblib.dump(ensemble_clone, os.path.join(MODEL_DIR, 'ensemble_clone.joblib'))
    joblib.dump(knn_fake, os.path.join(MODEL_DIR, 'knn_fake.joblib'))
    joblib.dump(knn_clone, os.path.join(MODEL_DIR, 'knn_clone.joblib'))
    
    # Save feature importances helper file
    fitted_rf_fake = ensemble_fake.named_estimators_['rf']
    importances = {FEATURE_COLS[i]: float(fitted_rf_fake.feature_importances_[i]) for i in range(len(FEATURE_COLS))}
    joblib.dump(importances, os.path.join(MODEL_DIR, 'feature_importances.joblib'))
    
    # Evaluate
    preds_fake = ensemble_fake.predict(X_test_scaled)
    preds_clone = ensemble_clone.predict(X_test_scaled)
    
    print("\n--- Fake Account Ensemble Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test_fake, preds_fake)*100:.2f}%")
    print(classification_report(y_test_fake, preds_fake))
    
    print("\n--- Clone Account Ensemble Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test_clone, preds_clone)*100:.2f}%")
    print(classification_report(y_test_clone, preds_clone))
    
    print("\nModels and preprocessors trained and saved successfully.")

def predict_profile(profile_data, scaler_dir=MODEL_DIR):
    """
    Production-ready real-time inference function.
    """
    scaler_path = os.path.join(scaler_dir, 'scaler.joblib')
    fake_path = os.path.join(scaler_dir, 'ensemble_fake.joblib')
    clone_path = os.path.join(scaler_dir, 'ensemble_clone.joblib')
    
    if not os.path.exists(scaler_path) or not os.path.exists(fake_path) or not os.path.exists(clone_path):
        raise FileNotFoundError("Models or preprocessors not found. Run train_realistic_models() first.")
        
    scaler = joblib.load(scaler_path)
    ensemble_fake = joblib.load(fake_path)
    ensemble_clone = joblib.load(clone_path)
    
    # Create single-record dataframe for feature engineering
    df = pd.DataFrame([profile_data])
    rename_map = {
        'tweets_count': 'posts_count',
        'followers_count': 'network_count',
        'duplicate_tweets': 'duplicate_posts',
        'name': 'display_name'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    if 'network_count' not in df.columns and 'followers_count' in profile_data:
        df['network_count'] = profile_data['followers_count']
    if 'posts_count' not in df.columns and 'posts_count' in profile_data:
        df['posts_count'] = profile_data['posts_count']
        
    df_feat = engineer_features(df)
    
    # Extract features matching columns
    X = df_feat[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    
    # Predict
    fake_prob = float(ensemble_fake.predict_proba(X_scaled)[0, 1])
    clone_prob = float(ensemble_clone.predict_proba(X_scaled)[0, 1])
    
    combined_risk = max(fake_prob, clone_prob) * 100
    
    if combined_risk < 30:
        classification = "Genuine"
    elif combined_risk < 60:
        classification = "Suspicious"
    else:
        classification = "Fake" if fake_prob > clone_prob else "Clone"
        
    return {
        'fake_probability': round(fake_prob, 4),
        'clone_probability': round(clone_prob, 4),
        'combined_risk_score': round(combined_risk, 1),
        'classification': classification
    }

if __name__ == '__main__':
    train_realistic_models()
    
    # Sample Test
    test_profile = {
        'username': 'fake_nikhita_983742',
        'followers_count': 5,
        'following_count': 3500,
        'account_age': 15,
        'posts_count': 850,
        'profile_picture': 0,
        'bio': '',
        'content_similarity': 0.85
    }
    print("\nSample Real-Time Inference on Suspicious Profile:")
    print(predict_profile(test_profile))
