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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Define feature columns
FEATURE_COLS = [
    'network_following_ratio',
    'username_digit_ratio',
    'username_has_trailing_digits',
    'profile_completeness',
    'is_new_and_aggressive',
    'account_age',
    'profile_picture',
    'post_frequency'
]

def generate_realistic_dataset(num_samples=6000):
    """
    Generates a balanced synthetic dataset representing social media accounts
    based on the exact behavioral rules for fake/clone accounts.
    """
    print(f"Generating realistic synthetic dataset with {num_samples} records...")
    np.random.seed(42)
    
    records = []
    
    # 1. Generate Genuine Accounts (50%)
    num_genuine = num_samples // 2
    for i in range(num_genuine):
        followers = int(np.random.lognormal(mean=9.5, sigma=1.5)) + 100
        following = int(np.random.lognormal(mean=8.5, sigma=1.5)) + 50
        age_days = int(np.random.uniform(90, 3650))
        posts = int(np.random.uniform(50, 6000))
        
        # Genuine features
        username = f"user_{np.random.randint(100, 999)}_{i}"
        bio = f"Self-proclaimed enthusiast of things. Creator."
        profile_pic = 1 if np.random.rand() < 0.98 else 0
        
        records.append({
            'username': username,
            'followers_count': followers,
            'following_count': following,
            'account_age': age_days,
            'posts_count': posts,
            'profile_picture': profile_pic,
            'bio': bio,
            'is_fake': 0,
            'is_clone': 0
        })
        
    # 2. Generate Fake/Bot Accounts (25%)
    num_fake = num_samples // 4
    for i in range(num_fake):
        # Fake profile details
        followers = int(np.random.uniform(0, 50))
        following = int(np.random.uniform(1000, 5000))  # Aggressive following
        age_days = int(np.random.uniform(1, 45))        # Often young
        
        # Rule 4: New and aggressive posts or following
        is_aggressive = np.random.rand() < 0.70
        if is_aggressive:
            posts = int(np.random.uniform(350, 1200))   # High posts for new account
        else:
            posts = int(np.random.uniform(0, 5))
            
        # Rule 3: Incomplete profile (no pic, empty bio)
        profile_pic = 0 if np.random.rand() < 0.85 else 1
        bio = "" if np.random.rand() < 0.90 else "Follow me!"
        
        # Rule 2: Trailing numbers pattern
        username = f"bot_scam_{np.random.randint(100000, 999999)}"
        
        records.append({
            'username': username,
            'followers_count': followers,
            'following_count': following,
            'account_age': age_days,
            'posts_count': posts,
            'profile_picture': profile_pic,
            'bio': bio,
            'is_fake': 1,
            'is_clone': 0
        })
        
    # 3. Generate Clone/Impersonator Accounts (25%)
    num_clones = num_samples // 4
    for i in range(num_clones):
        # Rule 1: Follower ratio < 0.1 (massive following, low followers)
        following = int(np.random.uniform(1500, 6000))
        followers = int(following * np.random.uniform(0.01, 0.09)) # guaranteed ratio < 0.1
        
        age_days = int(np.random.uniform(1, 60))        # Newly created
        posts = int(np.random.uniform(5, 60))
        profile_pic = 1 if np.random.rand() < 0.70 else 0 # Clones copy profile pic
        bio = "Official Account." if np.random.rand() < 0.5 else ""
        
        # Rule 2: High density of trailing numbers copying a name
        target_name = np.random.choice(["jack_sparrow", "elon_musk", "john_doe", "nikhita_gp"])
        username = f"{target_name}{np.random.randint(100000, 999999)}"
        
        records.append({
            'username': username,
            'followers_count': followers,
            'following_count': following,
            'account_age': age_days,
            'posts_count': posts,
            'profile_picture': profile_pic,
            'bio': bio,
            'is_fake': 0,
            'is_clone': 1
        })
        
    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def engineer_features(df):
    """
    Extracts engineered features matching the exact rules of fake/clone behavior.
    """
    df = df.copy()
    
    # Age conversion from days to years
    df['account_age'] = (df['account_age'] / 365.0).clip(lower=0.001)
    
    # 1. Follower-to-Following Ratio
    df['network_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
    
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
    # Age < 30 days is roughly < 0.082 years
    df['is_new_and_aggressive'] = (
        (df['account_age'] < 30) & 
        ((df['posts_count'] > 300) | (df['following_count'] > 1000))
    ).astype(int)
    
    # Post frequency (posts per day)
    df['post_frequency'] = df['posts_count'] / (df['account_age'] + 1)
    
    # Ensure final features are extracted
    return df

def train_realistic_models():
    print("=== SecureLock Model Pipeline Training ===")
    
    # Generate realistic data
    df_raw = generate_realistic_dataset(num_samples=50000)
    df_feat = engineer_features(df_raw)
    
    # Labels
    # is_suspicious = 1 if fake or clone, else 0
    y_fake = df_feat['is_fake']
    y_clone = df_feat['is_clone']
    
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
    rf_fake = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    xgb_fake = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, eval_metric='logloss')
    ensemble_fake = VotingClassifier(estimators=[('rf', rf_fake), ('xgb', xgb_fake)], voting='soft')
    ensemble_fake.fit(X_train_scaled, y_train_fake)
    
    # Clone Detector Ensemble
    rf_clone = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    xgb_clone = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, eval_metric='logloss')
    ensemble_clone = VotingClassifier(estimators=[('rf', rf_clone), ('xgb', xgb_clone)], voting='soft')
    ensemble_clone.fit(X_train_scaled, y_train_clone)
    
    # Train KNN models (n=7 for better smoothing and boundary classification)
    knn_fake = KNeighborsClassifier(n_neighbors=7)
    knn_fake.fit(X_train_scaled, y_train_fake)
    
    # Train KNN models
    knn_clone = KNeighborsClassifier(n_neighbors=7)
    knn_clone.fit(X_train_scaled, y_train_clone)
    
    # Save Ensembles and KNN models
    joblib.dump(ensemble_fake, os.path.join(MODEL_DIR, 'ensemble_fake.joblib'))
    joblib.dump(ensemble_clone, os.path.join(MODEL_DIR, 'ensemble_clone.joblib'))
    joblib.dump(knn_fake, os.path.join(MODEL_DIR, 'knn_fake.joblib'))
    joblib.dump(knn_clone, os.path.join(MODEL_DIR, 'knn_clone.joblib'))
    
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
    profile_data is a dict containing:
      - username (str)
      - followers_count (int)
      - following_count (int)
      - account_age (int/float - days)
      - posts_count (int)
      - profile_picture (int - 0 or 1)
      - bio (str)
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
        'bio': ''
    }
    print("\nSample Real-Time Inference on Suspicious Profile:")
    print(predict_profile(test_profile))
