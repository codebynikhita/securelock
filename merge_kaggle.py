import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'accounts.csv')

def merge_datasets():
    print("=== SecureLock Kaggle Dataset Merger ===")
    
    # 1. Load local dataset
    local_path = os.path.join(DATA_DIR, 'fake_clone_accounts.csv')
    if not os.path.exists(local_path):
        # Fall back to default location
        local_path = "/Users/nikhitagp/Downloads/Data clone/data/fake_clone_accounts.csv"
        
    print(f"Loading local dataset from: {local_path}")
    df_local = pd.read_csv(local_path)
    print(f"Loaded {len(df_local)} local records.")
    
    # 2. Download external Kaggle dataset from GitHub
    url = "https://raw.githubusercontent.com/jubins/MachineLearning-Detecting-Twitter-Bots/master/FinalProjectAndCode/kaggle_data/training_data_2_csv_UTF.csv"
    print(f"Downloading Kaggle dataset from: {url}")
    
    try:
        df_kag = pd.read_csv(url)
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        return
        
    print(f"Loaded {len(df_kag)} records from Kaggle Twitter Bot project.")
    
    # 3. Preprocess Kaggle columns to match local schema
    # Local columns: account_id,username,bio,profile_picture,account_age,tweets_count,followers_count,following_count,duplicate_tweets,content_similarity,follower_following_ratio,tweet_frequency,is_fake,is_clone,report,blocked,avg_distance,name
    
    # Clean created_at dates and calculate age in days
    if df_kag['created_at'].dtype == object:
        df_kag['created_at'] = df_kag['created_at'].str.replace('"', '').str.strip()
    ref_date = pd.to_datetime('2026-07-08')
    parsed_dates = pd.to_datetime(df_kag['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
    parsed_dates = parsed_dates.fillna(pd.to_datetime('2021-07-08'))
    df_kag['account_age'] = (ref_date - parsed_dates).dt.days
    
    # Map features
    df_kag['tweets_count'] = pd.to_numeric(df_kag['statuses_count'], errors='coerce').fillna(0).astype(int)
    df_kag['followers_count'] = pd.to_numeric(df_kag['followers_count'], errors='coerce').fillna(0).astype(int)
    df_kag['following_count'] = pd.to_numeric(df_kag['friends_count'], errors='coerce').fillna(0).astype(int)
    
    # Profile picture (inverse of default profile image)
    if df_kag['default_profile_image'].dtype == object:
        df_kag['default_profile_image'] = df_kag['default_profile_image'].str.replace('"', '').str.strip()
    def_img = pd.to_numeric(df_kag['default_profile_image'], errors='coerce').fillna(0)
    df_kag['profile_picture'] = np.where(def_img == 1, 0, 1)
    
    # Compute ratio and frequency
    df_kag['follower_following_ratio'] = df_kag['followers_count'] / (df_kag['following_count'] + 1)
    df_kag['tweet_frequency'] = df_kag['tweets_count'] / (df_kag['account_age'].clip(lower=1))
    
    # Missing columns filling
    df_kag['duplicate_tweets'] = 0
    df_kag['content_similarity'] = 0.1
    df_kag['is_fake'] = pd.to_numeric(df_kag['bot'], errors='coerce').fillna(0).astype(int)
    df_kag['is_clone'] = 0
    df_kag['report'] = 0
    df_kag['blocked'] = 0
    df_kag['avg_distance'] = 0.5
    
    # Clean username/name
    df_kag['username'] = df_kag['screen_name'].str.replace('"', '').str.strip()
    df_kag['name'] = df_kag['name'].str.replace('"', '').str.strip()
    df_kag['bio'] = df_kag['description'].str.replace('"', '').str.strip()
    df_kag['account_id'] = 'K' + df_kag['id_str'].astype(str).str.replace('"', '').str.strip()
    
    # Keep only target columns
    cols = [
        'account_id', 'username', 'bio', 'profile_picture', 'account_age', 
        'tweets_count', 'followers_count', 'following_count', 'duplicate_tweets', 
        'content_similarity', 'follower_following_ratio', 'tweet_frequency', 
        'is_fake', 'is_clone', 'report', 'blocked', 'avg_distance', 'name'
    ]
    
    df_kag_processed = df_kag[cols].copy()
    
    # 4. Merge datasets
    df_merged = pd.concat([df_local, df_kag_processed], ignore_index=True)
    # Remove duplicate usernames
    df_merged.drop_duplicates(subset=['username'], keep='first', inplace=True)
    
    # 5. Save merged dataset
    df_merged.to_csv(CSV_PATH, index=False)
    print(f"\n[MERGE SUCCESS] Merged dataset saved to {CSV_PATH} with {len(df_merged)} total records.")
    
if __name__ == '__main__':
    merge_datasets()
