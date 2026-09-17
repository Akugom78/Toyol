import os
import yaml
import bcrypt
from yaml.loader import SafeLoader

def manage_users():
    credentials_file = '.streamlit/credentials.yaml'
    
    if not os.path.exists(credentials_file):
        print("❌ Credentials file not found. Please run the app first to initialize users.")
        return
    
    with open(credentials_file, 'r') as file:
        credentials = yaml.load(file, Loader=SafeLoader)
    
    print("\n" + "="*60)
    print("ATC Knowledge Assistant - User Management")
    print("="*60)
    print("\nCurrent Users:")
    for username, user_data in credentials['credentials']['usernames'].items():
        print(f"  - {username}: {user_data['name']} ({user_data['email']})")
    
    print("\nOptions:")
    print("1. Add new user")
    print("2. Remove user")
    print("3. Change password")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        username = input("Username: ")
        name = input("Full name: ")
        email = input("Email: ")
        password = input("Password: ")
        
        # Hash the new password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        
        credentials['credentials']['usernames'][username] = {
            'email': email,
            'name': name,
            'password': hashed_password
        }
        
        with open(credentials_file, 'w') as file:
            yaml.dump(credentials, file, default_flow_style=False)
        
        print(f"✅ User '{username}' added successfully!")
    
    elif choice == '2':
        username = input("Username to remove: ")
        if username in credentials['credentials']['usernames']:
            del credentials['credentials']['usernames'][username]
            with open(credentials_file, 'w') as file:
                yaml.dump(credentials, file, default_flow_style=False)
            print(f"✅ User '{username}' removed successfully!")
        else:
            print(f"❌ User '{username}' not found!")
    
    elif choice == '3':
        username = input("Username: ")
        if username in credentials['credentials']['usernames']:
            new_password = input("New password: ")
            
            # Hash the new password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt).decode('utf-8')
            
            credentials['credentials']['usernames'][username]['password'] = hashed_password
            with open(credentials_file, 'w') as file:
                yaml.dump(credentials, file, default_flow_style=False)
            print(f"✅ Password changed successfully!")
        else:
            print(f"❌ User '{username}' not found!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    manage_users()