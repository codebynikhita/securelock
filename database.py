import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'securelock.db')

def get_db_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create Admins table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 2. Create Search Logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS search_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        display_name TEXT,
        platform TEXT NOT NULL,
        account_age REAL,
        followers INTEGER,
        following INTEGER,
        posts INTEGER,
        duplicate_posts INTEGER,
        content_similarity REAL,
        profile_picture INTEGER,
        fake_prob REAL,
        clone_prob REAL,
        risk_score REAL,
        classification TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 3. Create Reported Accounts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reported_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        platform TEXT NOT NULL,
        risk_score REAL,
        reason TEXT,
        reported_by TEXT DEFAULT 'User',
        status TEXT DEFAULT 'Pending',
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Seed default admin if table is empty
    cursor.execute("SELECT COUNT(*) FROM admins")
    if cursor.fetchone()[0] == 0:
        default_pwd_hash = generate_password_hash("admin123", method="pbkdf2:sha256")
        cursor.execute("INSERT INTO admins (username, password_hash) VALUES (?, ?)", ("admin", default_pwd_hash))
        print("Default admin created (username: admin, password: admin123)")
        
    conn.commit()
    conn.close()

# Admin Auth functions
def verify_admin(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE username = ?", (username,))
    admin = cursor.fetchone()
    conn.close()
    
    if admin and check_password_hash(admin['password_hash'], password):
        return True
    return False

# Logging functions
def log_search(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO search_logs (
        username, display_name, platform, account_age, followers, following, 
        posts, duplicate_posts, content_similarity, profile_picture, 
        fake_prob, clone_prob, risk_score, classification
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['username'],
        data.get('display_name', ''),
        data['platform'],
        data['account_age'],
        data['network_count'],
        data['following_count'],
        data['posts_count'],
        data.get('duplicate_posts', 0),
        data.get('content_similarity', 0.0),
        data.get('profile_picture', 1),
        data['fake_probability'],
        data['clone_probability'],
        data['combined_risk_score'],
        data['classification']
    ))
    conn.commit()
    conn.close()

def get_search_logs(limit=100):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM search_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return logs

def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total searches
    cursor.execute("SELECT COUNT(*) FROM search_logs")
    total_searches = cursor.fetchone()[0]
    
    # Get classifications distributions
    cursor.execute("SELECT classification, COUNT(*) FROM search_logs GROUP BY classification")
    class_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Ensure keys exist
    for c in ['Genuine', 'Suspicious', 'Fake']:
        if c not in class_dist:
            class_dist[c] = 0
            
    # Get platform distributions
    cursor.execute("SELECT platform, COUNT(*) FROM search_logs GROUP BY platform")
    platform_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Total reported
    cursor.execute("SELECT COUNT(*) FROM reported_accounts")
    total_reported = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_searches': total_searches,
        'class_distribution': class_dist,
        'platform_distribution': platform_dist,
        'total_reported': total_reported
    }

# Reported accounts functions
def report_account(username, platform, risk_score, reason):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO reported_accounts (username, platform, risk_score, reason)
    VALUES (?, ?, ?, ?)
    ''', (username, platform, risk_score, reason))
    conn.commit()
    conn.close()

def get_reported_accounts():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reported_accounts ORDER BY timestamp DESC")
    reports = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return reports

def update_report_status(report_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE reported_accounts SET status = ? WHERE id = ?", (status, report_id))
    conn.commit()
    conn.close()

def delete_report(report_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reported_accounts WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print("Database initialized.")
