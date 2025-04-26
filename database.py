import sqlite3
import os
import hashlib
import random
import string

class Database:
    def __init__(self, database='database.db', dir="./file"):
        self.connect = sqlite3.connect(database)
        self.cursor = self.connect.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON;")
        self.create_users_table()
        self.create_files_table()
        self.file_dir = dir
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

    def close(self):
        self.connect.close()

    # ---------------- USER MANAGEMENT--------------------
    def create_users_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        self.connect.commit()
    
    def generate_unique_user_id(self, length=8):
        chars = string.ascii_letters + string.digits
        while(True):
            user_id = ''.join(random.choices(chars, k=length))
            self.cursor.execute('''
                SELECT * FROM users 
                WHERE user_id = ?
            ''', (user_id,))
            if not self.cursor.fetchone():
                return user_id

    def create_user(self, username, password):
        user_id = self.generate_unique_user_id()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        try:
            self.cursor.execute('''
                INSERT INTO users (user_id, username, password) VALUES (?, ?, ?)
            ''', (user_id, username, hashed_password))
            self.connect.commit()
            print(f"[INFO] User '{username}' created successfully.")
            return True
        except sqlite3.IntegrityError:
            print(f"[ERROR] Username '{username}' already exists.")
            return False

    def login(self, username, password):
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute('''
            SELECT user_id FROM users 
            WHERE username = ? AND password = ?
        ''', (username, hashed_password))
        result = self.cursor.fetchone()
        if result:
            print(f"[INFO] User '{username}' logged in successfully")
            return result[0]
        else:
            print('[ERROR] Invalid username or password')
            return None
        

    # ---------------- FILE MANAGEMENT--------------------
    def create_files_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                display_name TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        ''')
        self.connect.commit()
    
    # generate unique file name to store
    def generate_unique_file_name(self, length=10):
        chars = string.ascii_letters + string.digits
        if length > len(chars):
            raise ValueError("Length exceeds number of unique characters available.")
        while(True):
            rand_str = ''.join(random.sample(chars, length))
            rand_path = os.path.join(self.file_dir, rand_str)
            self.cursor.execute('''
                SELECT * FROM files
                WHERE file_path = ?
            ''', (rand_path,))
            if not self.cursor.fetchone():
                return rand_str

    # insert to files table
    def insert_user_file(self, user_id, file_path, display_name):
        if not os.path.isfile(file_path):
            print(f"[ERROR] File '{file_path}' does not exist or is not a file")
            return
        if display_name is None:
            display_name = os.path.basename(file_path)
        self.cursor.execute('''
            INSERT INTO files (user_id, file_path, display_name) VALUES (?, ?, ?)
        ''', (user_id, file_path, display_name))
        self.connect.commit()
        print(f"[INFO] Inserted file '{display_name}' to user with id: '{user_id}' successfully.")

        # create chatlog to the new file
        last_row_id = self.cursor.lastrowid
        self.create_chatlog_table(str(last_row_id))
    
    # insert a file in binary to the table
    def insert_user_file_binary(self, user_id, file_binary, file_name):
        saved_file_name = self.generate_unique_file_name()
        save_path = os.path.join(self.file_dir, saved_file_name)
        with open(save_path, 'wb') as f:
            f.write(file_binary)
        self.insert_user_file(user_id, os.path.abspath(save_path), file_name)

    # return all files of a user
    def get_user_files(self, user_id):
        self.cursor.execute('''
            SELECT id, display_name FROM files
            WHERE user_id = ?
        ''', (user_id,))
        result = self.cursor.fetchall()
        return [{"id": file_id, "name": file_name} for (file_id, file_name) in result]
    
    # get file with id
    def get_file_by_id(self, id):
        self.cursor.execute('''
            SELECT file_path, display_name from files
            WHERE id = ?
        ''', (id,))
        result = self.cursor.fetchone()
        if result:
            print(f"[INFO] Retrieved file with id: {id}")
            return result
        else:
            return None

    # create chatlog for new file
    def create_chatlog_table(self, file_id):
        table_name = "chatlog_" + file_id
        self.cursor.execute(f'''
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role = 'user' OR role = 'assistant'),
                text TEXT NOT NULL
            )
        ''')
        self.connect.commit()
        print(f"[INFO] Created table: {table_name}")

    # get chat history
    def get_history(self, file_id):
        table_name = "chatlog_" + str(file_id)
        self.cursor.execute(f'''
            SELECT role, text
            FROM (
                SELECT * 
                FROM {table_name}
                ORDER BY id DESC
                LIMIT 100
            ) query
            ORDER BY id ASC
        ''')
        self.connect.commit()
        rows = self.cursor.fetchall()
        result = [{"role": role, "content": text} for role, text in rows]
        return result
    
    # save chat message
    def log_chat(self, id, text, role):
        table_name = "chatlog_" + str(id)
        self.cursor.execute(f'''
            INSERT INTO {table_name} (role, text) VALUES ('{role}', '{text}')
        ''')
        self.connect.commit()