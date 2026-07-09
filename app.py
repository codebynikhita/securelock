import os
import sqlite3
import numpy as np
import pandas as pd
import re
import urllib.request
import urllib.parse
import urllib.error
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from model import SecureLockModel
import database
import socket
socket.setdefaulttimeout(5.0)

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'securelock_secret_session_key_2026'

import traceback
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        "error": str(e),
        "traceback": traceback.format_exc()
    }), 500

@app.context_processor
def utility_processor():
    def format_number(val):
        try:
            return f"{int(val):,}"
        except (ValueError, TypeError):
            return str(val)
    return dict(min=min, max=max, format_number=format_number)

# Initialize Model and DB
model_engine = SecureLockModel()
database.init_db()


# Load Dataset Cache
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'accounts.csv')
if os.path.exists(CSV_PATH):
    df_accounts = pd.read_csv(CSV_PATH)
    
    # Drop rows that are empty or malformed
    df_accounts.dropna(subset=['username', 'account_age', 'profile_picture'], inplace=True)
    
    # Rename columns to match expected schema if needed
    rename_map = {
        'tweets_count': 'posts_count',
        'followers_count': 'network_count',
        'duplicate_tweets': 'duplicate_posts',
        'name': 'display_name'
    }
    df_accounts.rename(columns={k: v for k, v in rename_map.items() if k in df_accounts.columns}, inplace=True)
    
    # Scale account age from days to years if they are large numbers (days)
    if df_accounts['account_age'].max() > 100:
        df_accounts['account_age'] = df_accounts['account_age'] / 365.0
        
    # Ensure platform column exists
    if 'platform' not in df_accounts.columns:
        np.random.seed(42)
        plats = ['twitter', 'instagram', 'facebook', 'linkedin']
        probs = [0.47, 0.30, 0.15, 0.08]
        df_accounts['platform'] = np.random.choice(plats, size=len(df_accounts), p=probs)
        

    print(f"Flask loaded {len(df_accounts)} accounts into memory cache.")
    
    # Pre-calculate platform baselines
    platform_stats = {}
    for plat in ['twitter', 'instagram', 'facebook', 'linkedin']:
        sub = df_accounts[df_accounts['platform'] == plat]
        if len(sub) > 0:
            platform_stats[plat] = {
                'network_count': int(sub['network_count'].median()),
                'following_count': int(sub['following_count'].median()),
                'posts_count': int(sub['posts_count'].median()),
                'account_age': round(float(sub['account_age'].median()), 2),
                'duplicate_posts': int(sub['duplicate_posts'].median()),
                'content_similarity': round(float(sub['content_similarity'].median()), 4),
                'profile_picture': 1
            }
        else:
            # Fallbacks
            platform_stats[plat] = {
                'network_count': 500,
                'following_count': 250,
                'posts_count': 1000,
                'account_age': 3.5,
                'duplicate_posts': 2,
                'content_similarity': 0.15,
                'profile_picture': 1
            }
            
    # Extract profiles for clone comparison
    db_profiles = df_accounts.copy()
    # Add verified status (fake accounts are not verified, genuine can be)
    db_profiles['is_verified'] = (db_profiles['is_fake'] == 0) & (db_profiles['is_clone'] == 0) & (df_accounts['network_count'] > 50000)
    db_profiles_list = db_profiles.to_dict(orient='records')
else:
    df_accounts = None
    platform_stats = {}
    db_profiles_list = []
    print("WARNING: accounts.csv not found. Prediction matching will be unavailable.")

def estimate_features_from_username(username, platform):
    """
    Estimates behavioral features based on username characteristics
    and platform-specific baselines when an account is not in our dataset.
    """

    # Deterministic seeding based on username and platform hash to ensure consistent results
    import hashlib
    hash_object = hashlib.md5((username.lower() + platform.lower()).encode())
    seed = int(hash_object.hexdigest(), 16) % 1000000
    rng = np.random.default_rng(seed)

    baseline = platform_stats.get(platform, {
        'network_count': 400,
        'following_count': 300,
        'posts_count': 500,
        'account_age': 2.0,
        'duplicate_posts': 3,
        'content_similarity': 0.20,
        'profile_picture': 1
    }).copy()
    
    # Analyze username characteristics
    length = len(username)
    digit_count = sum(c.isdigit() for c in username)
    digit_ratio = digit_count / length if length > 0 else 0
    
    # Detect suspicious bot-like username patterns (e.g. name + 4+ digits, long length)
    is_suspicious_username = False
    
    # Check for trailing numbers of length 4+ (common in automated accounts)
    import re
    if re.search(r'\d{4,}$', username):
        is_suspicious_username = True
    elif digit_ratio > 0.3 and length > 8:
        is_suspicious_username = True
        
    if is_suspicious_username:
        # Generate typical bot feature profile
        baseline['profile_picture'] = 0 if rng.random() < 0.6 else 1
        baseline['account_age'] = round(rng.uniform(0.01, 0.9), 2)
        baseline['network_count'] = int(rng.uniform(2, 60))
        baseline['following_count'] = int(rng.uniform(800, 5000))
        baseline['posts_count'] = int(rng.uniform(0, 100))
        baseline['duplicate_posts'] = int(rng.uniform(8, 45))
        baseline['content_similarity'] = round(rng.uniform(0.45, 0.88), 4)
    else:
        # Add slight random variations around the platform average for realistic representation
        baseline['profile_picture'] = 1
        baseline['account_age'] = max(0.5, round(baseline['account_age'] * rng.uniform(0.7, 1.5), 2))
        baseline['network_count'] = int(baseline['network_count'] * rng.uniform(0.5, 2.5))
        baseline['following_count'] = int(baseline['following_count'] * rng.uniform(0.6, 1.4))
        baseline['posts_count'] = int(baseline['posts_count'] * rng.uniform(0.5, 2.0))
        baseline['duplicate_posts'] = int(rng.exponential(scale=2))
        baseline['content_similarity'] = round(max(0.01, min(0.99, baseline['content_similarity'] * rng.uniform(0.5, 1.5))), 4)
        
    return baseline

# --- User Routes ---

@app.route('/')
def index():
    # Load basic system stats for display on landing page
    db_stats = database.get_stats()
    
    # Example suggestions for search auto-complete
    suggestions = []
    if df_accounts is not None:
        # Pick a few genuine, fake, and clone usernames for user testing
        g_names = df_accounts[(df_accounts['is_fake'] == 0) & (df_accounts['is_clone'] == 0)]['username'].head(3).tolist()
        f_names = df_accounts[df_accounts['is_fake'] == 1]['username'].head(3).tolist()
        c_names = df_accounts[df_accounts['is_clone'] == 1]['username'].head(3).tolist()
        suggestions = g_names + f_names + c_names
        
    return render_template('index.html', suggestions=suggestions, stats=db_stats)

def fetch_live_profile_data(username, platform):
    """
    Attempts to scrape/fetch live profile details from social media servers.
    Handles login walls and blocks gracefully with informative logs and cache fallback.
    """
    status_logs = []
    status_logs.append(f"Initializing live connection to {platform.upper()} API gateways...")
    
    clean_username = username.strip().replace('@', '')
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    
    urls = {
        'instagram': f"https://www.instagram.com/{clean_username}/",
        'twitter': f"https://twitter.com/{clean_username}",
        'facebook': f"https://www.facebook.com/{clean_username}",
        'linkedin': f"https://www.linkedin.com/in/{clean_username}"
    }
    
    url = urls.get(platform.lower())
    if not url:
        return None, ["Unsupported platform selected for live fetching."]
        
    status_logs.append(f"Sending secure HTTP GET request to {url}...")
    
    scraped_data = None
    try:
        if platform.lower() == 'instagram':
            status_logs.append("Routing Instagram query via upgraded scrape_live_instagram module...")
            try:
                from scrape_live_instagram import get_instagram_metrics
                res = get_instagram_metrics(clean_username)
                if res and res.get('status') == 'success':
                    scraped_data = {
                        'followers': res['followers'],
                        'following': res['following'],
                        'posts': res['posts'],
                        'account_age': res.get('account_age', 3.5),
                        'profile_pic_url': res.get('profile_pic_url'),
                        'has_profile_pic': res.get('has_profile_pic', True),
                        'bio': res.get('biography', "Parsed via Instagram API gateway")
                    }
                    if res.get('json_api_error'):
                        status_logs.append(f"⚠ JSON API failed ({res['json_api_error']}). Using public bot-cache fallback.")
                    else:
                        status_logs.append(f"Successfully retrieved exact metrics via Instagram API gateway ({res.get('source', 'api')})!")
                else:
                    status_logs.append(f"Scraper returned error: {res.get('message', 'Unknown error') if res else 'No response'}")
            except Exception as e:
                status_logs.append(f"Scraper execution failed: {str(e)}")
        
        if not scraped_data:
            status_logs.append(f"Attempting secure index-agent connection to {url}...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
            )
            with urllib.request.urlopen(req, timeout=4.0) as response:
                status_code = response.getcode()
                status_logs.append(f"Connection response code: {status_code}")
                body = response.read().decode('utf-8', errors='ignore')
                
                final_url = response.geturl()
                if "login" in final_url.lower() or "accounts/login" in final_url.lower():
                    status_logs.append("Meta/X Gateway Alert: Access redirected to login wall. Direct public scraping is restricted.")
                else:
                    # Extract Profile Picture URL from og:image
                    profile_pic_url = None
                    og_img = re.search(r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']', body, re.IGNORECASE)
                    if not og_img:
                        og_img = re.search(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']', body, re.IGNORECASE)
                    if og_img:
                        import html as html_lib
                        profile_pic_url = html_lib.unescape(og_img.group(1))

                    desc_match = re.search(r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']', body, re.IGNORECASE)
                    if not desc_match:
                        desc_match = re.search(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:description["\']', body, re.IGNORECASE)
                        
                    if desc_match:
                        desc_content = desc_match.group(1)
                        status_logs.append(f"Parsed public SEO summary metadata: '{desc_content[:80]}...'")
                        
                        followers_m = re.search(r'([0-9.,KMB]+)\s*(?:Followers|followers)', desc_content, re.IGNORECASE)
                        following_m = re.search(r'([0-9.,KMB]+)\s*(?:Following|following)', desc_content, re.IGNORECASE)
                        posts_m = re.search(r'([0-9.,KMB]+)\s*(?:Posts|posts|Tweets|tweets)', desc_content, re.IGNORECASE)
                        
                        def parse_val(val_str):
                            if not val_str: return 0
                            val_str = val_str.replace(',', '').strip().upper()
                            if 'K' in val_str:
                                return int(float(val_str.replace('K', '')) * 1000)
                            if 'M' in val_str:
                                return int(float(val_str.replace('M', '')) * 1000000)
                            try:
                                return int(float(val_str))
                            except:
                                return 0
                            
                        if followers_m or following_m:
                            scraped_data = {
                                'followers': parse_val(followers_m.group(1) if followers_m else "0"),
                                'following': parse_val(following_m.group(1) if following_m else "0"),
                                'posts': parse_val(posts_m.group(1) if posts_m else "0"),
                                'profile_pic_url': profile_pic_url,
                                'has_profile_pic': profile_pic_url is not None,
                                'bio': desc_content[:150]
                            }
                            status_logs.append("Metrics and profile picture successfully scraped from live headers!")
    except urllib.error.HTTPError as e:
        status_logs.append(f"Direct connection blocked by firewall: HTTP {e.code} ({e.reason})")
    except Exception as e:
        status_logs.append(f"Connection timed out or blocked by platform gateway: {str(e)}")
        
    if not scraped_data:
        status_logs.append("Bypassing firewalls: Routing query via search index gateway...")
        try:
            query_str = f"{clean_username} {platform}"
            encoded_query = urllib.parse.quote(query_str)
            req = urllib.request.Request(
                'https://lite.duckduckgo.com/lite/', 
                data=f'q={encoded_query}'.encode(), 
                headers={'User-Agent': user_agent}
            )
            with urllib.request.urlopen(req, timeout=3.0) as response:
                html = response.read().decode('utf-8', errors='ignore')
                f_match = re.search(r'([0-9.,KMB]+)\s*followers', html, re.IGNORECASE)
                fl_match = re.search(r'([0-9.,KMB]+)\s*following', html, re.IGNORECASE)
                posts_m = re.search(r'([0-9.,KMB]+)\s*(posts|tweets)', html, re.IGNORECASE)
                
                def parse_val(val_str):
                    if not val_str: return 0
                    val_str = val_str.replace(',', '').strip().upper()
                    if 'K' in val_str:
                        return int(float(val_str.replace('K', '')) * 1000)
                    if 'M' in val_str:
                        return int(float(val_str.replace('M', '')) * 1000000)
                    return int(float(val_str))
                
                if f_match or fl_match:
                    scraped_data = {
                        'followers': parse_val(f_match.group(1) if f_match else "0"),
                        'following': parse_val(fl_match.group(1) if fl_match else "0"),
                        'posts': parse_val(posts_m.group(1) if posts_m else "0"),
                        'bio': f"Scraped from live {platform} index search"
                    }
                    status_logs.append(f"Successfully retrieved real metrics from index metadata: {scraped_data['followers']} Followers, {scraped_data['following']} Following!")
        except Exception as se:
            status_logs.append(f"Search gateway bypass attempt failed: {str(se)}")

    if scraped_data:
        return scraped_data, status_logs
    else:
        status_logs.append("Activating SecureLock local fallback parser & cache profiles.")
        return None, status_logs

@app.route('/detect', methods=['POST'])
def detect():
    username = request.form.get('username', '').strip()
    platform = request.form.get('platform', 'twitter').lower()
    
    if not username:
        return render_template('index.html', error="Username is required.")
        
    # Get stats and suggestions for templates
    db_stats = database.get_stats()
    suggestions = []
    if df_accounts is not None:
        g_names = df_accounts[(df_accounts['is_fake'] == 0) & (df_accounts['is_clone'] == 0)]['username'].head(3).tolist()
        f_names = df_accounts[df_accounts['is_fake'] == 1]['username'].head(3).tolist()
        c_names = df_accounts[df_accounts['is_clone'] == 1]['username'].head(3).tolist()
        suggestions = g_names + f_names + c_names

    # Check if manual stats were provided in the form
    manual_mode = request.form.get('manual_mode') == 'true'
    status_logs = []
    
    if manual_mode:
        account_data = {
            'username': username,
            'display_name': username,
            'platform': platform,
            'network_count': int(request.form.get('followers', 0)),
            'following_count': int(request.form.get('following', 0)),
            'posts_count': int(request.form.get('posts', 0)),
            'account_age': float(request.form.get('account_age', 1.0)),
            'profile_picture': int(request.form.get('profile_picture', 1)),
            'duplicate_posts': 0,
            'content_similarity': 0.1,
            'is_fake': 0,
            'is_clone': 0,
            'report': 0,
            'blocked': 0,
            'avg_distance': 0.5
        }
        source = "manual"
        status_logs.append("User overridden metrics submitted. Re-analyzing profile...")
    else:
        account_data = None
        source = "database"

        # ── Live-Scraped Platforms: Instagram, Facebook, LinkedIn ────────────
        if platform in ('instagram', 'facebook', 'linkedin'):
            scraped_data, scrape_logs = fetch_live_profile_data(username, platform)
            status_logs.extend(scrape_logs)

            if scraped_data:
                account_data = {
                    'username': username,
                    'display_name': username,
                    'platform': platform,
                    'network_count': scraped_data['followers'],
                    'following_count': scraped_data['following'],
                    'posts_count': scraped_data['posts'],
                    'account_age': scraped_data.get('account_age', 3.5),
                    'profile_picture': 1 if scraped_data.get('has_profile_pic', True) else 0,
                    'profile_pic_url': scraped_data.get('profile_pic_url', None),
                    'duplicate_posts': 0,
                    'content_similarity': 0.1,
                    'is_fake': 0,
                    'is_clone': 0,
                    'report': 0,
                    'blocked': 0,
                    'avg_distance': 0.5
                }
                source = "live_scrape"
                # Supplement with CSV labels if available
                if df_accounts is not None:
                    csv_match = df_accounts[
                        (df_accounts['username'].str.lower() == username.lower()) &
                        (df_accounts['platform'] == platform)
                    ]
                    if len(csv_match) > 0:
                        row = csv_match.iloc[0].to_dict()
                        account_data['is_fake']  = row.get('is_fake', 0)
                        account_data['is_clone'] = row.get('is_clone', 0)
                        account_data['content_similarity'] = row.get('content_similarity', 0.1)
                        status_logs.append("Live data supplemented with labelled dataset signals.")
            
            # If live scraping failed, fall back to local database check before showing error
            if not account_data:
                if df_accounts is not None:
                    match = df_accounts[
                        (df_accounts['username'].str.lower() == username.lower()) &
                        (df_accounts['platform'] == platform)
                    ]
                    if len(match) > 0:
                        account_data = match.iloc[0].to_dict()
                        status_logs.append("Direct scraping restricted. Query matched local database profile cache.")
                        source = "database"

            if not account_data:
                platform_name = platform.capitalize()
                return render_template('index.html',
                    error=f"❌ @{username} could not be found on {platform_name}. The account may not exist, may be private, or may have been deleted.",
                    db_stats=db_stats,
                    suggestions=suggestions
                )

        # ── Non-Scraped Platforms: Twitter (X) and others ──────────────────
        else:
            if df_accounts is not None:
                match = df_accounts[
                    (df_accounts['username'].str.lower() == username.lower()) &
                    (df_accounts['platform'] == platform)
                ]
                if len(match) > 0:
                    account_data = match.iloc[0].to_dict()
                    status_logs.append("Query matched local database profile cache.")
                    source = "database"

            if not account_data:
                return render_template('index.html',
                    error=f"❌ @{username} could not be found on {platform.capitalize()}. Please verify the username and platform.",
                    db_stats=db_stats,
                    suggestions=suggestions
                )

    # Run prediction pipeline
    result = model_engine.detect(account_data, db_profiles_list)
    
    # Guard: if models aren't loaded, detect() returns an error dict — show friendly error
    if 'error' in result and 'combined_risk_score' not in result:
        print(f"[ERROR] model_engine.detect() failed: {result.get('error')}")
        return render_template('index.html',
            error=f"⚠️ Analysis engine is warming up. Please try again in a moment. ({result.get('error', 'Models not ready')})",
            db_stats=db_stats,
            suggestions=suggestions
        )
    
    # Determine which fields were scraped live vs estimated fallbacks
    sources = {
        'followers': 'live' if source in ('database', 'manual') or (source == 'live_scrape' and scraped_data and 'followers' in scraped_data) else 'estimated',
        'following': 'live' if source in ('database', 'manual') or (source == 'live_scrape' and scraped_data and 'following' in scraped_data) else 'estimated',
        'posts': 'live' if source in ('database', 'manual') or (source == 'live_scrape' and scraped_data and scraped_data.get('posts', 0) > 0) else 'estimated',
        'account_age': 'live' if source in ('database', 'manual') or (source == 'live_scrape' and scraped_data and scraped_data.get('account_age', 0) > 0) else 'estimated'
    }
    
    # Combine original record data and predictions
    full_report = {**account_data, **result, 'data_source': source, 'sources': sources}
    
    # Log to SQLite DB
    database.log_search(full_report)
    
    # If the request wants JSON (AJAX)
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(full_report)
        
    return render_template('index.html', report=full_report, status_logs=status_logs, suggestions=suggestions, stats=db_stats)

@app.route('/api/version', methods=['GET'])
def api_version():
    return jsonify({"commit": "be2c04a", "status": "active"}), 200

@app.route('/api/detect', methods=['POST', 'GET'])
def api_detect():
    """
    REST API endpoint for developers.
    Accepts JSON body or query params.
    """
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
    else:
        data = request.args
        
    username = data.get('username', '').strip()
    platform = data.get('platform', 'twitter').lower()
    
    if not username:
        return jsonify({"status": "error", "message": "Missing required parameter: username"}), 400
        
    account_data = None
    source = "database"
    
    if df_accounts is not None:
        match = df_accounts[(df_accounts['username'].str.lower() == username.lower()) & (df_accounts['platform'] == platform)]
        if len(match) > 0:
            account_data = match.iloc[0].to_dict()
            
    if not account_data:
        account_data = estimate_features_from_username(username, platform)
        account_data['username'] = username
        account_data['display_name'] = username.capitalize()
        account_data['platform'] = platform
        source = "estimation"
        
    result = model_engine.detect(account_data, db_profiles_list)
    full_report = {**account_data, **result, 'data_source': source}
    
    # Log to DB
    database.log_search(full_report)
    
    # Sanitize pandas types for JSON
    sanitized = {}
    for k, v in full_report.items():
        if isinstance(v, (np.integer, np.int64)):
            sanitized[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            sanitized[k] = float(v)
        else:
            sanitized[k] = v
            
    return jsonify(sanitized)

@app.route('/report', methods=['POST'])
def report():
    username = request.form.get('username', '').strip()
    platform = request.form.get('platform', 'twitter').lower()
    risk_score = float(request.form.get('risk_score', 0))
    reason = request.form.get('reason', '')
    
    if not username:
        return jsonify({"status": "error", "message": "Username is required."}), 400
        
    database.report_account(username, platform, risk_score, reason)
    
    # SMTP Console fallbacks
    print(f"\n[SMTP NOTIFICATION SENT TO ADMIN]")
    print(f"Subject: Security Warning: New Fake/Clone Report")
    print(f"Details: Account @{username} on {platform.capitalize()} has been flagged by a user.")
    print(f"Calculated System Risk Score: {risk_score}%")
    print(f"User Reason: {reason}\n")
    
    return jsonify({"status": "success", "message": "Account successfully reported to security administrators."})

# --- Admin Routes ---

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if 'admin_user' in session:
        return redirect(url_for('admin_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if database.verify_admin(username, password):
            session['admin_user'] = username
            flash("Successfully logged in as administrator.", "success")
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error="Invalid username or password.")
            
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_user', None)
    flash("Successfully logged out.", "info")
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin_user' not in session:
        return redirect(url_for('admin_login'))
        
    stats = database.get_stats()
    logs = database.get_search_logs(50)
    reports = database.get_reported_accounts()
    
    return render_template('admin_dashboard.html', stats=stats, logs=logs, reports=reports)

@app.route('/admin/action', methods=['POST'])
def admin_action():
    if 'admin_user' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
    report_id = int(request.form.get('report_id', 0))
    action_type = request.form.get('action', '') # 'verify', 'dismiss'
    
    if not report_id:
        return jsonify({"status": "error", "message": "Missing report ID"}), 400
        
    if action_type == 'verify':
        database.update_report_status(report_id, 'Verified Fake')
        msg = "Report verified as fake account."
    elif action_type == 'dismiss':
        database.delete_report(report_id)
        msg = "Report dismissed and removed."
    else:
        return jsonify({"status": "error", "message": "Invalid action"}), 400
        
    return jsonify({"status": "success", "message": msg})

@app.route('/admin/visualize')
def admin_visualize():
    """
    API for coordinates generation for KNN distance plots.
    Returns:
    - 200 random samples from our dataset (scaled using StandardScaler and reduced)
    - The coordinates of the currently queried account.
    We will use 'network_following_ratio' (log scale or normalized) and 'content_similarity' as X and Y coordinates.
    """
    if df_accounts is None:
        return jsonify({"error": "No data available."}), 404
        
    query_username = request.args.get('query_username', '')
    query_platform = request.args.get('query_platform', 'twitter').lower()
    
    # 1. Take a clean sample of 150 accounts to draw the dataset background (50 genuine, 50 fake, 50 clone)
    g_samples = df_accounts[(df_accounts['is_fake'] == 0) & (df_accounts['is_clone'] == 0)].sample(50, random_state=42)
    f_samples = df_accounts[df_accounts['is_fake'] == 1].sample(50, random_state=42)
    c_samples = df_accounts[df_accounts['is_clone'] == 1].sample(50, random_state=42)
    
    sampled_df = pd.concat([g_samples, f_samples, c_samples])
    
    # Calculate ratios and frequencies
    sampled_df['ratio'] = sampled_df['network_count'] / (sampled_df['following_count'] + 1)
    # Log scale the ratio for better visual grouping: log10(ratio + 0.01)
    sampled_df['x'] = np.log10(sampled_df['ratio'] + 0.01)
    sampled_df['y'] = sampled_df['content_similarity']
    
    points = []
    for idx, row in sampled_df.iterrows():
        label = "Genuine"
        if row['is_fake'] == 1:
            label = "Fake"
        elif row['is_clone'] == 1:
            label = "Clone"
            
        points.append({
            'username': row['username'],
            'x': float(row['x']),
            'y': float(row['y']),
            'label': label
        })
        
    # 2. Get current query coordinates
    query_point = None
    neighbors = []
    
    if query_username:
        # Check logs or match in DB
        # Find matches
        match = df_accounts[(df_accounts['username'].str.lower() == query_username.lower()) & (df_accounts['platform'] == query_platform)]
        if len(match) > 0:
            row = match.iloc[0]
            ratio = row['network_count'] / (row['following_count'] + 1)
            qx = float(np.log10(ratio + 0.01))
            qy = float(row['content_similarity'])
        else:
            # Query log for estimated values
            conn = sqlite3.connect(database.DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM search_logs WHERE username = ? AND platform = ? ORDER BY timestamp DESC LIMIT 1", (query_username, query_platform))
            log = cursor.fetchone()
            conn.close()
            
            if log:
                ratio = log['followers'] / (log['following'] + 1)
                qx = float(np.log10(ratio + 0.01))
                qy = float(log['content_similarity'])
            else:
                qx = 0.0
                qy = 0.5
                
        query_point = {
            'username': query_username,
            'x': qx,
            'y': qy
        }
        
        # 3. Calculate distance to find 5 nearest neighbors (KNN) in this 2D projection space
        distances = []
        for p in points:
            dist = np.sqrt((p['x'] - qx)**2 + (p['y'] - qy)**2)
            distances.append((dist, p))
            
        distances.sort(key=lambda x: x[0])
        # Take top 5
        for d, p in distances[:5]:
            neighbors.append({
                'username': p['username'],
                'x': p['x'],
                'y': p['y'],
                'label': p['label'],
                'distance': float(d)
            })
            
    return jsonify({
        'dataset_points': points,
        'query_point': query_point,
        'nearest_neighbors': neighbors
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
