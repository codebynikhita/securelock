import os
import unittest
from model_def import XGBClassifier
from app import app
import database

class SecureLockIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        
        # Ensure database is initialized
        database.init_db()
        
    def test_landing_page(self):
        """Test that index page loads successfully."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'SecureLock', res.data)
        
    def test_api_detect_genuine(self):
        """Test API endpoint with a verified/genuine username from dataset."""
        # Narendra Modi is in our celebrity group in synthetic data (first 1500 accounts)
        # We can query Narendra Modi or look for another name from CSV
        res = self.client.post('/api/detect', data={
            'username': 'susanrivera',
            'platform': 'twitter'
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['username'], 'susanrivera')
        self.assertEqual(data['classification'], 'Genuine')
        self.assertLess(data['combined_risk_score'], 50)
        
    def test_api_detect_fake_estimation(self):
        """Test API endpoint with a bot-like username (gets estimated as Fake)."""
        res = self.client.post('/api/detect', data={
            'username': 'bot_account_9827351',
            'platform': 'twitter'
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['classification'], 'Fake')
        self.assertGreaterEqual(data['combined_risk_score'], 60)
        self.assertTrue(len(data['explanations']) > 0)
        
    def test_report_account(self):
        """Test reporting an account."""
        res = self.client.post('/report', data={
            'username': 'test_scammer',
            'platform': 'instagram',
            'risk_score': 85.5,
            'reason': 'Spamming direct messages'
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        
    def test_admin_login_fail(self):
        """Test admin login with bad credentials."""
        res = self.client.post('/admin/login', data={
            'username': 'admin',
            'password': 'wrongpassword'
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Invalid username or password', res.data)

if __name__ == '__main__':
    unittest.main()
