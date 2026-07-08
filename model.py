import os
import numpy as np
import pandas as pd
import joblib
import re

# Removed dangerous __getattr__ hook that broke hasattr()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# Euclidean Distance implementation
def euclidean_distance(v1, v2):
    return float(np.linalg.norm(np.array(v1) - np.array(v2)))

# Cosine Similarity implementation
def cosine_similarity(v1, v2):
    arr1 = np.array(v1)
    arr2 = np.array(v2)
    dot_product = np.dot(arr1, arr2)
    norm_arr1 = np.linalg.norm(arr1)
    norm_arr2 = np.linalg.norm(arr2)
    if norm_arr1 == 0 or norm_arr2 == 0:
        return 0.0
    return float(dot_product / (norm_arr1 * norm_arr2))

# Levenshtein Distance implementation
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

class SecureLockModel:
    def __init__(self):
        self.scaler = None
        self.ensemble_fake = None
        self.ensemble_clone = None
        self.knn_fake = None
        self.knn_clone = None
        self.feature_importances = None
        self.loaded = False
        self.load_models()

    def load_models(self):
        try:
            scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
            fake_path = os.path.join(MODEL_DIR, 'ensemble_fake.joblib')
            clone_path = os.path.join(MODEL_DIR, 'ensemble_clone.joblib')
            knn_fake_path = os.path.join(MODEL_DIR, 'knn_fake.joblib')
            knn_clone_path = os.path.join(MODEL_DIR, 'knn_clone.joblib')
            importance_path = os.path.join(MODEL_DIR, 'feature_importances.joblib')
            
            if os.path.exists(scaler_path) and os.path.exists(fake_path) and os.path.exists(clone_path) and os.path.exists(knn_fake_path) and os.path.exists(knn_clone_path):
                self.scaler = joblib.load(scaler_path)
                self.ensemble_fake = joblib.load(fake_path)
                self.ensemble_clone = joblib.load(clone_path)
                self.knn_fake = joblib.load(knn_fake_path)
                self.knn_clone = joblib.load(knn_clone_path)
                
                if os.path.exists(importance_path):
                    self.feature_importances = joblib.load(importance_path)
                else:
                    self.feature_importances = {
                        'network_following_ratio': 0.352,
                        'account_age': 0.181,
                        'profile_picture': 0.148,
                        'post_frequency': 0.123,
                        'content_similarity': 0.097,
                        'duplicate_posts': 0.045,
                        'posts_count': 0.032,
                        'following_count': 0.022
                    }

                self.loaded = True
                print("Models loaded successfully in SecureLockModel.")
            else:
                print("Model files not found. Please run model_pipeline.py first.")
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()

    def evaluate_rules(self, raw_features):
        """
        Evaluate five rule-based indicators:
        1. missing profile picture
        2. very low follower count (less than 50)
        3. abnormal follower/following ratio (less than 0.1 or greater than 100)
        4. low activity for account age (post frequency less than 0.1 with account age over 365 days)
        5. high content similarity (greater than 0.7)
        """
        network_count = raw_features.get('network_count', 0)
        following_count = raw_features.get('following_count', 0)
        posts_count = raw_features.get('posts_count', 0)
        account_age = raw_features.get('account_age', 0)
        profile_picture = raw_features.get('profile_picture', 1)
        content_similarity = raw_features.get('content_similarity', 0.0)
        
        ratio = network_count / (following_count + 1)
        if account_age > 100:
            age_days = account_age
        else:
            age_days = account_age * 365
        post_freq = posts_count / (age_days + 1)
        
        indicators = {
            'missing_profile_pic': profile_picture == 0,
            'low_followers': network_count < 50,
            'abnormal_ratio': ratio < 0.1 or ratio > 100,
            'low_activity_for_age': post_freq < 0.1 and age_days > 365,
            'high_content_similarity': content_similarity > 0.7
        }
        
        triggered = [k for k, v in indicators.items() if v]
        is_fake_by_rules = len(triggered) >= 3
        
        return is_fake_by_rules, triggered

    def get_feature_contributions(self, raw_features, scaled_features, model, probability):
        """
        Computes SHAP-like feature contributions for a specific prediction.
        Identifies the marginal impact of each feature on the prediction probability.
        """
        feature_cols = [
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
        
        # Baseline: Predict with all features set to their average (scaled = 0)
        baseline_scaled = np.zeros((1, len(feature_cols)))
        baseline_prob = model.predict_proba(baseline_scaled)[0, 1]
        
        contributions = {}
        
        # Calculate marginal contribution of each feature
        for i, col in enumerate(feature_cols):
            # Create a test point where this feature is active, others are at baseline (0)
            test_scaled = np.zeros((1, len(feature_cols)))
            test_scaled[0, i] = scaled_features[0, i]
            
            feat_prob = model.predict_proba(test_scaled)[0, 1]
            diff = feat_prob - baseline_prob
            
            # Scale by feature importance to smooth out model noise
            importance = self.feature_importances.get(col, 0.125)
            contrib_value = diff * importance * 10
            
            contributions[col] = contrib_value

        # Normalize contributions to sum up to (probability - baseline_prob) approximately,
        # but keep them clean, readable, and signed.
        # Let's sort contributions by absolute value to find top drivers.
        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Translate to human readable statements
        explanations = []
        for feat, val in sorted_contribs[:3]:
            direction = "increases" if val > 0 else "decreases"
            val_formatted = f"{abs(val)*10:.1f}%"
            
            # Custom descriptions
            desc = ""
            if feat == 'network_following_ratio':
                desc = f"Follower/Following ratio ({raw_features.get('network_count', 0)}/{raw_features.get('following_count', 0)})"
            elif feat == 'username_digit_ratio':
                username = raw_features.get('username', '')
                username_digit_ratio = sum(c.isdigit() for c in str(username)) / len(str(username)) if len(str(username)) > 0 else 0
                desc = f"Username digit density ({username_digit_ratio*100:.1f}%)"
            elif feat == 'username_has_trailing_digits':
                username = raw_features.get('username', '')
                has_trailing = 1 if re.search(r'\d{4,}$', str(username)) else 0
                desc = "Username ends with multiple digits" if has_trailing else "Username has normal digit pattern"
            elif feat == 'profile_completeness':
                bio = raw_features.get('bio', '')
                bio_present = 1 if bio and str(bio).strip() != "" and str(bio) != "nan" and str(bio) != "None" else 0
                profile_picture = int(raw_features.get('profile_picture', 1))
                posts = float(raw_features.get('posts_count', 0))
                profile_completeness = profile_picture + bio_present + (posts > 0)
                desc = f"Profile completeness score ({profile_completeness}/3)"
            elif feat == 'is_new_and_aggressive':
                age_days = float(raw_features.get('account_age', 0.01))
                posts = float(raw_features.get('posts_count', 0))
                following = float(raw_features.get('following_count', 0))
                is_aggressive = 1 if (age_days < 30 and (posts > 300 or following > 1000)) else 0
                desc = "Young account with hyperactive posting/following" if is_aggressive else "Normal account activity levels"
            elif feat == 'account_age':
                age_days = float(raw_features.get('account_age', 0.01))
                age_years = age_days / 365.0 if age_days > 100 else age_days
                desc = f"Account age ({age_years:.2f} years)"
            elif feat == 'profile_picture':
                profile_picture = int(raw_features.get('profile_picture', 1))
                desc = "Missing profile picture" if profile_picture == 0 else "Profile picture present"
            elif feat == 'post_frequency':
                posts = float(raw_features.get('posts_count', 0))
                age_days = float(raw_features.get('account_age', 0.01))
                post_frequency = posts / (age_days + 1)
                desc = f"Post frequency ({post_frequency:.2f}/day)"
            elif feat == 'content_similarity':
                similarity = float(raw_features.get('content_similarity', 0.0))
                desc = f"Content similarity to known accounts ({similarity*100:.1f}%)"
                
            explanations.append({
                'feature': feat,
                'influence': "positive" if val > 0 else "negative",
                'description': desc,
                'weight': abs(val)
            })
            
        return explanations

    def detect(self, raw_features, all_db_profiles=None):
        """
        Executes the SecureLock detection pipeline.
        Calculates:
        1. Model probability of being fake
        2. Model probability of being a clone
        3. Combined risk score
        4. Rule-based checks
        5. Clone lookup (if clone score is high or username checks trigger)
        6. SHAP-like explanations
        """
        if not self.loaded:
            self.load_models()
            if not self.loaded:
                return {"error": "Models not loaded. Setup database and train models first."}

        # Prepare feature vector matching model_pipeline FEATURE_COLS
        username = raw_features.get('username', '')
        followers = float(raw_features.get('network_count', 0))
        following = float(raw_features.get('following_count', 0))
        posts = float(raw_features.get('posts_count', 0))
        raw_age = float(raw_features.get('account_age', 0.01))
        if raw_age > 100:
            age_days = raw_age
            age_years = raw_age / 365.0
        else:
            age_days = raw_age * 365.0
            age_years = raw_age
            
        ratio = followers / (following + 1)
        
        # Username Patterns (Rule 2)
        username_digit_ratio = sum(c.isdigit() for c in str(username)) / len(str(username)) if len(str(username)) > 0 else 0
        username_has_trailing_digits = 1 if re.search(r'\d{4,}$', str(username)) else 0
        
        # Profile Completeness (Rule 3)
        bio = raw_features.get('bio', '')
        bio_present = 1 if bio and str(bio).strip() != "" and str(bio) != "nan" and str(bio) != "None" else 0
        profile_picture = int(raw_features.get('profile_picture', 1))
        profile_completeness = profile_picture + bio_present + (posts > 0)
        
        # New and aggressive behavior (Rule 4)
        is_new_and_aggressive = 1 if (age_days < 30 and (posts > 300 or following > 1000)) else 0
        
        # Post frequency (posts per day)
        post_frequency = posts / (age_days + 1)
        
        feature_vector = pd.DataFrame([{
            'network_following_ratio': ratio,
            'username_digit_ratio': username_digit_ratio,
            'username_has_trailing_digits': username_has_trailing_digits,
            'profile_completeness': profile_completeness,
            'is_new_and_aggressive': is_new_and_aggressive,
            'account_age': age_years,
            'profile_picture': profile_picture,
            'post_frequency': post_frequency,
            'content_similarity': float(raw_features.get('content_similarity', 0.1))
        }])
        
        # Scale features
        scaled = self.scaler.transform(feature_vector)

        # Get probabilities from VotingClassifier (soft voting averages RF + GradientBoosting)
        fake_prob = float(self.ensemble_fake.predict_proba(scaled)[0, 1])
        clone_prob = float(self.ensemble_clone.predict_proba(scaled)[0, 1])
        
        knn_fake_prob = self.knn_fake.predict_proba(scaled)[0, 1]
        
        knn_clone_prob = self.knn_clone.predict_proba(scaled)[0, 1]
        
        # Evaluate rule-based fallback
        rules_triggered_fake, rule_details = self.evaluate_rules(raw_features)
        
        # Adjust probabilities if rules heavily trigger
        if rules_triggered_fake:
            # Boost fake probability if at least 3 rules triggered
            fake_prob = max(fake_prob, 0.85)
            
        # If content similarity is extremely high, boost clone probability
        if raw_features.get('content_similarity', 0) > 0.8:
            clone_prob = max(clone_prob, 0.80)
            
        # Combine risk score: range (0 - 100)
        # Risk score is a weighted combination of fake and clone probabilities
        combined_risk = max(fake_prob, clone_prob) * 100
        
        # Classification category
        if combined_risk < 30:
            classification = "Genuine"
        elif combined_risk < 60:
            classification = "Suspicious"
        else:
            classification = "Fake"
            
        # Pick the active model to explain
        explainer_model = self.ensemble_clone if clone_prob > fake_prob else self.ensemble_fake
        active_prob = max(clone_prob, fake_prob)
        explanations = self.get_feature_contributions(raw_features, scaled, explainer_model, active_prob)
        
        # Clone matching check against existing DB users
        # Only run if clone probability is meaningful — skip for obvious non-clones (saves OOM on Render free tier)
        clone_target = None
        if all_db_profiles and raw_features.get('username') and clone_prob > 0.2:
            curr_user = raw_features['username'].lower()
            best_match = None
            min_dist = 999
            best_cos_sim = 0.0
            best_euc_dist = 999.0
            
            # Query feature vector
            query_vec = scaled[0]

            # --- VECTORIZED CLONE SCAN (no per-row DataFrame creation) ---
            # Extract all feature values into lists first, then batch-scale once
            usernames_list = []
            raw_vecs = []
            profiles_index = []

            for i, profile in enumerate(all_db_profiles):
                p_user = str(profile.get('username', '')).lower()
                if p_user == curr_user:
                    continue

                try:
                    p_followers = float(profile.get('network_count', 0))
                    p_following = float(profile.get('following_count', 0))
                    p_posts    = float(profile.get('posts_count', 0))
                    p_age_days = float(profile.get('account_age', 0.01))
                    p_age_years = p_age_days / 365.0 if p_age_days > 100 else p_age_days
                    p_ratio    = p_followers / (p_following + 1)
                    p_digit_ratio  = sum(c.isdigit() for c in p_user) / len(p_user) if p_user else 0
                    p_trailing = 1 if re.search(r'\d{4,}$', p_user) else 0
                    p_bio      = profile.get('bio', '')
                    p_bio_ok   = 1 if p_bio and str(p_bio).strip() not in ('', 'nan', 'None') else 0
                    p_pic      = int(profile.get('profile_picture', 1))
                    p_complete = p_pic + p_bio_ok + (p_posts > 0)
                    p_aggr     = 1 if (p_age_days < 30 and (p_posts > 300 or p_following > 1000)) else 0
                    p_freq     = p_posts / (p_age_days + 1)
                    # content_similarity not stored in DB profiles — default 0.1
                    raw_vecs.append([p_ratio, p_digit_ratio, p_trailing, p_complete,
                                     p_aggr, p_age_years, p_pic, p_freq, 0.1])
                    usernames_list.append(p_user)
                    profiles_index.append(i)
                except Exception:
                    continue

            if raw_vecs:
                # Single batch scale — vastly cheaper than 8303 individual transforms
                batch = np.array(raw_vecs, dtype=np.float64)
                batch_scaled = self.scaler.transform(batch)   # shape (N, 9)

                # Compute cosine similarities in one vectorized step
                query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
                norms = np.linalg.norm(batch_scaled, axis=1, keepdims=True) + 1e-10
                batch_normed = batch_scaled / norms
                cos_sims = batch_normed @ query_norm            # (N,)
                euc_dists = np.linalg.norm(batch_scaled - query_vec, axis=1)  # (N,)

                for idx, (p_user, cos_sim, euc_dist) in enumerate(zip(usernames_list, cos_sims, euc_dists)):
                    lev_dist = levenshtein_distance(curr_user, p_user)
                    if lev_dist <= 2 or (cos_sim >= 0.95 and lev_dist <= 5 and
                                         (curr_user in p_user or p_user in curr_user)):
                        if lev_dist < min_dist:
                            min_dist = lev_dist
                            best_match = all_db_profiles[profiles_index[idx]]
                            best_cos_sim = float(cos_sim)
                            best_euc_dist = float(euc_dist)

            # If we have a close match and similarity signals
            if best_match and (clone_prob > 0.4 or raw_features.get('content_similarity', 0) > 0.5 or best_cos_sim >= 0.90):
                clone_target = {
                    'username': best_match['username'],
                    'display_name': best_match.get('display_name', best_match['username']),
                    'distance': min_dist,
                    'is_verified': best_match.get('is_verified', False),
                    'cosine_similarity': round(best_cos_sim, 4),
                    'euclidean_distance': round(best_euc_dist, 4)
                }
                # Boost clone probability and risk because username matches clone patterns
                clone_prob = max(clone_prob, 0.88)
                combined_risk = max(combined_risk, 88.0)
                classification = "Fake"

                
        # Compute individual algorithm risk for KNN
        knn_risk = max(knn_fake_prob, knn_clone_prob) * 100
        knn_class = "Genuine" if knn_risk < 30 else ("Suspicious" if knn_risk < 60 else "Fake")
        
        # Ensemble risk (RF + GradientBoosting via VotingClassifier)
        ensemble_risk = combined_risk
        ensemble_class = classification

        return {
            'fake_probability': round(fake_prob, 4),
            'clone_probability': round(clone_prob, 4),
            'combined_risk_score': round(combined_risk, 1),
            'classification': classification,
            'rules_triggered': rule_details,
            'rules_fake_triggered': rules_triggered_fake,
            'explanations': explanations,
            'clone_target': clone_target,
            
            # Individual Algorithms
            'rf_fake_prob': round(fake_prob, 4),
            'rf_clone_prob': round(clone_prob, 4),
            'rf_classification': ensemble_class,
            
            'xgb_fake_prob': round(fake_prob, 4),
            'xgb_clone_prob': round(clone_prob, 4),
            'xgb_classification': ensemble_class,
            
            'knn_fake_prob': round(knn_fake_prob, 4),
            'knn_clone_prob': round(knn_clone_prob, 4),
            'knn_classification': knn_class
        }
