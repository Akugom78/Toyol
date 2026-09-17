import os
import yaml
import bcrypt
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

class UserManager:
    def __init__(self):
        self.credentials_file = '.streamlit/credentials.yaml'
        
    def initialize_users(self):
        """Initialize default users if credentials file doesn't exist AND no secrets are provided"""
        if not os.path.exists(self.credentials_file):
            # If running on Streamlit Cloud, st.secrets will have the credentials
            if hasattr(st, 'secrets') and 'credentials' in st.secrets:
                return  # Secrets exist, do not create a local default file
                
            # Local fallback: create default admin
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
        """Load user credentials from Streamlit Secrets (Cloud) or local file"""
        # 1. Try to load from Streamlit Cloud Secrets first (TOML format)
        if hasattr(st, 'secrets') and 'credentials' in st.secrets:
            return {
                'credentials': {
                    'usernames': {
                        user: {
                            'email': st.secrets['credentials']['usernames'][user]['email'],
                            'name': st.secrets['credentials']['usernames'][user]['name'],
                            'password': st.secrets['credentials']['usernames'][user]['password']
                        }
                        for user in st.secrets['credentials']['usernames']
                    }
                },
                'cookie': {
                    'expiry_days': st.secrets['cookie']['expiry_days'],
                    'key': st.secrets['cookie']['key'],
                    'name': st.secrets['cookie']['name']
                }
            }
            
        # 2. Fallback to local file (YAML format)
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