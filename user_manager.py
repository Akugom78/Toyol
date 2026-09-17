import os
import yaml
import bcrypt
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

class UserManager:
    def __init__(self):
        self.credentials_file = '.streamlit/credentials.yaml'
        
    def initialize_users(self):
        """Initialize default users if credentials file doesn't exist"""
        if not os.path.exists(self.credentials_file):
            # Hash the default password securely using bcrypt
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw('admin123'.encode('utf-8'), salt).decode('utf-8')
            
            credentials = {
                'credentials': {
                    'usernames': {
                        'admin': {
                            'email': 'admin@atc-assistant.com',
                            'name': 'Administrator',
                            'password': hashed_password
                        }
                    }
                },
                'cookie': {
                    'expiry_days': 30,
                    'key': 'atc_assistant_secret_key_1234567890_abcdef',
                    'name': 'atc_assistant_cookie'
                }
            }
            
            os.makedirs('.streamlit', exist_ok=True)
            with open(self.credentials_file, 'w') as file:
                yaml.dump(credentials, file, default_flow_style=False)
                
    def load_credentials(self):
        """Load user credentials"""
        with open(self.credentials_file, 'r') as file:
            return yaml.load(file, Loader=SafeLoader)
            
    def get_authenticator(self):
        """Create and return authenticator"""
        self.initialize_users()
        credentials = self.load_credentials()
        
        authenticator = stauth.Authenticate(
            credentials['credentials'],
            credentials['cookie']['name'],
            credentials['cookie']['key'],
            credentials['cookie']['expiry_days']
        )
        
        return authenticator