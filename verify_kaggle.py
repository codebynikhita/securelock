import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_def import XGBClassifier # Ensure namespace fallback works

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

def run_kaggle_validation():
    print("=== SecureLock Kaggle Generalization Validation ===")
    
    # 1. Load external Kaggle dataset from GitHub
    url = "https://raw.githubusercontent.com/jubins/MachineLearning-Detecting-Twitter-Bots/master/FinalProjectAndCode/kaggle_data/training_data_2_csv_UTF.csv"
    print(f"Downloading dataset from: {url}")
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error downloading: {e}")
        return
        
    print(f"Loaded {len(df)} external records from Kaggle Twitter Bot project.")
    
    # Clean string fields
    if df['created_at'].dtype == object:
        df['created_at'] = df['created_at'].str.replace('"', '').str.strip()
        
    # Parse dates with pandas (timezone-naive to prevent subtracting offset-naive from offset-aware)
    ref_date = pd.to_datetime('2026-07-08')
    parsed_dates = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
    # Fill NaNs with a default date (e.g. 5 years ago)
    parsed_dates = parsed_dates.fillna(pd.to_datetime('2021-07-08'))
    
    # Calculate account age in years
    df['account_age'] = (ref_date - parsed_dates).dt.days / 365.0
    # Ensure no zero/negative age
    df['account_age'] = df['account_age'].clip(lower=0.01)
    
    # Map features
    df['posts_count'] = pd.to_numeric(df['statuses_count'], errors='coerce').fillna(0)
    df['network_count'] = pd.to_numeric(df['followers_count'], errors='coerce').fillna(0)
    df['following_count'] = pd.to_numeric(df['friends_count'], errors='coerce').fillna(0)
    
    # Profile picture is 0 if default_profile_image is True/1, else 1
    # Handle possible string quotes around values
    if df['default_profile_image'].dtype == object:
        df['default_profile_image'] = df['default_profile_image'].str.replace('"', '').str.strip()
    def_img = pd.to_numeric(df['default_profile_image'], errors='coerce').fillna(0)
    df['profile_picture'] = np.where(def_img == 1, 0, 1)
    
    # Since Kaggle datasets don't have text duplicates/similarity metrics precomputed,
    # we fill with standard baseline values
    df['duplicate_posts'] = 0
    df['content_similarity'] = 0.1
    
    # Feature Engineering
    df['network_following_ratio'] = df['network_count'] / (df['following_count'] + 1)
    df['post_frequency'] = df['posts_count'] / (df['account_age'] * 365 + 1)
    
    feature_cols = [
        'network_following_ratio',
        'account_age',
        'profile_picture',
        'post_frequency',
        'content_similarity',
        'duplicate_posts',
        'posts_count',
        'following_count'
    ]
    
    # 2. Extract features and target label
    # Kaggle dataset has label 'bot' (1 = Bot/Fake, 0 = Genuine)
    df['bot'] = pd.to_numeric(df['bot'], errors='coerce').fillna(0).astype(int)
    
    X = df[feature_cols].copy()
    y = df['bot']
    
    # 3. Load pre-trained scaler and fake account ensemble
    scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
    model_path = os.path.join(MODEL_DIR, 'ensemble_fake.joblib')
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("Pretrained models not found! Run train.py first.")
        return
        
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    preds = model.predict(X_scaled)
    
    # Evaluate
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    
    print("\nEvaluation Results on Kaggle Dataset:")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    
    print("\n[SUCCESS] Pre-trained SecureLock models evaluated successfully on external Kaggle dataset.")

if __name__ == '__main__':
    run_kaggle_validation()
