import hashlib
import os
import random
import sqlite3
import string

from models.qna import PDFProcessor, read_pdf
from models.summarize import summarize
from utils.pdf_utils import get_pdf_page_count
from models.get_citation import get_list_of_urls

database_path = "sqlite_db/database.db"
storing_dir = os.path.join("..", "..", "file")  # Outside of `src` folder


class Database:
    def __init__(self, database=database_path, storing_dir=storing_dir):
        self.connect = sqlite3.connect(database)
        self.cursor = self.connect.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON;")
        self.create_users_table()
        self.create_files_table()
        self.pdf_extract = PDFProcessor()
        self.file_dir = storing_dir
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.connect.close()

    # ---------------- USER MANAGEMENT--------------------
    def create_users_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        """
        )
        self.connect.commit()

    def generate_unique_user_id(self, length=8):
        chars = string.ascii_letters + string.digits
        while True:
            user_id = "".join(random.choices(chars, k=length))
            self.cursor.execute(
                """
                SELECT * FROM users 
                WHERE user_id = ?
            """,
                (user_id,),
            )
            if not self.cursor.fetchone():
                return user_id

    def create_user(self, username, password):
        user_id = self.generate_unique_user_id()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        try:
            self.cursor.execute(
                """
                INSERT INTO users (user_id, username, password) VALUES (?, ?, ?)
            """,
                (user_id, username, hashed_password),
            )
            self.connect.commit()
            print(f"[INFO] User '{username}' created successfully.")
            return True
        except sqlite3.IntegrityError:
            print(f"[ERROR] Username '{username}' already exists.")
            return False

    def login(self, username, password):
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute(
            """
            SELECT user_id FROM users 
            WHERE username = ? AND password = ?
        """,
            (username, hashed_password),
        )
        result = self.cursor.fetchone()
        if result:
            print(f"[INFO] User '{username}' logged in successfully")
            return result[0]
        else:
            print("[ERROR] Invalid username or password")
            return None

    # ---------------- FILE MANAGEMENT--------------------
    def create_files_table(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                display_name TEXT NOT NULL,
                summarize TEXT NOT NULL,
                num INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """
        )
        self.connect.commit()

    # generate unique file name to store
    def generate_unique_file_name(self, length=10):
        chars = string.ascii_letters + string.digits
        if length > len(chars):
            raise ValueError("Length exceeds number of unique characters available.")
        while True:
            rand_str = "".join(random.sample(chars, length))
            rand_path = os.path.join(self.file_dir, rand_str)
            self.cursor.execute(
                """
                SELECT * FROM files
                WHERE file_path = ?
            """,
                (rand_path,),
            )
            if not self.cursor.fetchone():
                return rand_str + ".pdf"

    # insert to files table
    def insert_user_file(self, file_id, user_id, file_path, display_name, summarize, page_num):
        if not os.path.isfile(file_path):
            print(f"[ERROR] File '{file_path}' does not exist or is not a file")
            return
        if display_name is None:
            display_name = os.path.basename(file_path)
        self.cursor.execute(
            """
            INSERT INTO files (id, user_id, file_path, display_name, summarize, num) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (file_id, user_id, file_path, display_name, summarize, page_num),
        )
        self.connect.commit()
        print(
            f"[INFO] Inserted file '{display_name}' to user with id: '{user_id}' successfully."
        )
        # create chatlog to the new file
        self.create_chatlog_table(file_id)

    # insert a file in binary to the table
    def insert_user_file_binary(self, user_id, file_binary, file_name):
        saved_file_name = self.generate_unique_file_name()
        save_path = os.path.join(self.file_dir, saved_file_name)
        with open(save_path, "wb") as f:
            f.write(file_binary)
        smr = summarize(self.pdf_extract.extract_text(save_path))
        # smr = "Temporary summary"  # Placeholder for the actual summary
        file_id = read_pdf(save_path)
        import logging
        logging.info(f"[INFO] File ID: {file_id}")
        page_num = get_pdf_page_count(save_path)
        self.insert_user_file(file_id, user_id, save_path, file_name, smr, page_num)
        self.create_citation_table(file_id, save_path)

    def citation_table(self, file_id):
        return "citation_" + hashlib.sha256(file_id.encode()).hexdigest()

    def create_citation_table(self, file_id, path):
        cites = get_list_of_urls(path)
        table_name = self.citation_table(file_id)
        self.cursor.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT NOT NULL
            )
        """
        )
        for cite in cites:
            self.cursor.execute(
                f"INSERT INTO {table_name} (title, url) VALUES (?, ?)",
                (cite["title"], cite["url"])
            )
        self.connect.commit()

    # return all files of a user
    def get_user_files(self, user_id):
        self.cursor.execute(
            """
            SELECT id, display_name, num FROM files
            WHERE user_id = ?
        """,
            (user_id,),
        )
        result = self.cursor.fetchall()
        file_list = [{"id": file_id, "name": file_name, "num": page_num} for (file_id, file_name, page_num) in result]
        for file in file_list:
            self.cursor.execute(
                f"""SELECT title, url FROM {self.citation_table(file["id"])}"""
            )
            self.connect.commit()
            cites = self.cursor.fetchall()
            file["cite"] = [{"title": title, "url": url} for title, url in cites]
        return file_list


    # get file with id
    def get_file_by_id(self, id):
        self.cursor.execute(
            """
            SELECT file_path, display_name from files
            WHERE id = ?
        """,
            (id,),
        )
        result = self.cursor.fetchone()
        if result:
            print(f"[INFO] Retrieved file with id: {id}")
            return result
        else:
            return None

    def get_smr_by_id(self, id):
        self.cursor.execute(
            """
            SELECT summarize from files
            WHERE id = ?
        """,
            (id,),
        )
        result = self.cursor.fetchone()
        if result:
            print(f"[INFO] Retrieved summary with id: {id}")
            return result
        else:
            return None

    def chatlog_table(self, file_id):
        return "chat_" + hashlib.sha256(file_id.encode()).hexdigest()
    # create chatlog for new file
    def create_chatlog_table(self, file_id):
        table_name = self.chatlog_table(file_id)
        self.cursor.execute(
            f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role = 'user' OR role = 'assistant'),
                text TEXT NOT NULL
            )
        """
        )
        self.connect.commit()
        print(f"[INFO] Created table: {table_name}")

    # get chat history
    def get_history(self, file_id):
        table_name = self.chatlog_table(file_id)
        self.cursor.execute(
            f"""
            SELECT role, text
            FROM (
                SELECT * 
                FROM {table_name}
                ORDER BY id DESC
                LIMIT 100
            ) query
            ORDER BY id ASC
        """
        )
        self.connect.commit()
        rows = self.cursor.fetchall()
        result = [{"role": role, "content": text} for role, text in rows]
        return result

    # save chat message
    def log_chat(self, file_id, text, role):
        table_name = self.chatlog_table(file_id)
        self.cursor.execute(
            f"""
            INSERT INTO {table_name} (role, text) VALUES ('{role}', '{text}')
        """
        )
        self.connect.commit()
