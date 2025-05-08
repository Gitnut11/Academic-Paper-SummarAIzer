import hashlib
import os
import random
import sqlite3
import string
import logging

from models.get_citation import get_list_of_urls
from models.qna import PDFProcessor, read_pdf
from models.summarize import summarize
from utils.pdf_utils import get_pdf_page_count

# Configure logging
logger = logging.getLogger(__name__)

# Database and file storage paths
database_path = "database/database.db"
storing_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "files"
)  # Outside of `src` folder


class Database:
    """
    A class that handles user management, PDF file storage, summarization, citation extraction,
    and chat logging using an SQLite database.
    """
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
        """Closes the database connection."""
        self.connect.close()

    # ---------------- USER MANAGEMENT --------------------
    def create_users_table(self):
        """Creates the users table if it does not exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        """)
        self.connect.commit()

    def generate_unique_user_id(self, length=8):
        """
        Generates a unique alphanumeric user ID.

        Args:
            length (int): Desired length of the user ID.

        Returns:
            str: A unique user ID.
        """
        chars = string.ascii_letters + string.digits
        while True:
            user_id = "".join(random.choices(chars, k=length))
            self.cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            if not self.cursor.fetchone():
                return user_id

    def create_user(self, username, password):
        """
        Registers a new user.

        Args:
            username (str): The user's name.
            password (str): The user's password.

        Returns:
            bool: True if user creation is successful, False otherwise.
        """
        user_id = self.generate_unique_user_id()
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        try:
            self.cursor.execute("""
                INSERT INTO users (user_id, username, password)
                VALUES (?, ?, ?)
            """, (user_id, username, hashed_password))
            self.connect.commit()
            logger.info(f"User '{username}' created successfully.")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Username '{username}' already exists.")
            return False

    def login(self, username, password):
        """
        Logs a user in by verifying credentials.

        Args:
            username (str): The user's name.
            password (str): The user's password.

        Returns:
            str | None: The user ID if login is successful, None otherwise.
        """
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        self.cursor.execute("""
            SELECT user_id FROM users 
            WHERE username = ? AND password = ?
        """, (username, hashed_password))
        result = self.cursor.fetchone()
        if result:
            logger.info(f"User '{username}' logged in successfully.")
            return result[0]
        else:
            logger.warning("Invalid username or password.")
            return None

    # ---------------- FILE MANAGEMENT --------------------
    def create_files_table(self):
        """Creates the files table if it does not exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                display_name TEXT NOT NULL,
                summarize TEXT NOT NULL,
                num INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)
        self.connect.commit()

    def generate_unique_file_name(self, length=10):
        """
        Generates a unique file name for saving uploaded PDFs.

        Args:
            length (int): Desired length of the file name (without extension).

        Returns:
            str: A unique file name ending with .pdf
        """
        chars = string.ascii_letters + string.digits
        if length > len(chars):
            raise ValueError("Length exceeds number of unique characters available.")
        while True:
            rand_str = "".join(random.sample(chars, length))
            rand_path = os.path.join(self.file_dir, rand_str)
            self.cursor.execute("SELECT * FROM files WHERE file_path = ?", (rand_path,))
            if not self.cursor.fetchone():
                return rand_str + ".pdf"

    def insert_user_file(self, file_id, user_id, file_path, display_name, summarize, page_num):
        """
        Inserts file metadata into the files table.

        Args:
            file_id (str): Unique file identifier.
            user_id (str): ID of the owner user.
            file_path (str): Path where the file is saved.
            display_name (str): Name shown to the user.
            summarize (str): Summary of the PDF.
            page_num (int): Number of pages in the PDF.
        """
        if not os.path.isfile(file_path):
            logger.warning(f"File '{file_path}' does not exist or is not a file.")
            return

        if display_name is None:
            display_name = os.path.basename(file_path)

        self.cursor.execute("""
            INSERT INTO files (id, user_id, file_path, display_name, summarize, num)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (file_id, user_id, file_path, display_name, summarize, page_num))
        self.connect.commit()
        logger.info(f"Inserted file '{display_name}' for user '{user_id}'.")
        self.create_chatlog_table(file_id)

    def insert_user_file_binary(self, user_id, file_binary, file_name):
        """
        Saves binary PDF data to disk and inserts metadata into the database.

        Args:
            user_id (str): ID of the user uploading the file.
            file_binary (bytes): File contents.
            file_name (str): Original name of the file.
        """
        saved_file_name = self.generate_unique_file_name()
        save_path = os.path.join(self.file_dir, saved_file_name)
        with open(save_path, "wb") as f:
            f.write(file_binary)

        smr = summarize(self.pdf_extract.extract_text(save_path))
        file_id = read_pdf(save_path)
        page_num = get_pdf_page_count(save_path)

        self.insert_user_file(file_id, user_id, save_path, file_name, smr, page_num)
        self.create_citation_table(file_id, save_path)

    def citation_table(self, file_id):
        """Generates a unique citation table name based on file ID."""
        return "citation_" + hashlib.sha256(file_id.encode()).hexdigest()

    def create_citation_table(self, file_id, path):
        """
        Creates and populates a table of citations for a given file.

        Args:
            file_id (str): Unique file ID.
            path (str): File path of the PDF.
        """
        cites = get_list_of_urls(path)
        table_name = self.citation_table(file_id)
        self.cursor.execute(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT NOT NULL
            )
        """)
        for cite in cites:
            self.cursor.execute(
                f"INSERT INTO {table_name} (title, url) VALUES (?, ?)",
                (cite["title"], cite["url"])
            )
        self.connect.commit()

    def get_user_files(self, user_id):
        """
        Retrieves all files for a given user along with citation data.

        Args:
            user_id (str): User's ID.

        Returns:
            list: List of file records with citations.
        """
        self.cursor.execute("""
            SELECT id, display_name, num FROM files
            WHERE user_id = ?
        """, (user_id,))
        result = self.cursor.fetchall()

        file_list = [{"id": file_id, "name": file_name, "num": page_num}
                     for file_id, file_name, page_num in result]

        for file in file_list:
            self.cursor.execute(f"SELECT title, url FROM {self.citation_table(file['id'])}")
            cites = self.cursor.fetchall()
            file["cite"] = [{"title": title, "url": url} for title, url in cites]

        return file_list

    def get_file_by_id(self, id):
        """
        Retrieves the file path and display name for a given file ID.

        Args:
            id (str): File ID.

        Returns:
            tuple | None: (file_path, display_name) or None if not found.
        """
        self.cursor.execute("""
            SELECT file_path, display_name FROM files
            WHERE id = ?
        """, (id,))
        result = self.cursor.fetchone()
        if result:
            logger.info(f"Retrieved file with ID: {id}")
            return result
        return None

    def get_smr_by_id(self, id):
        """
        Retrieves the summary for a given file ID.

        Args:
            id (str): File ID.

        Returns:
            str | None: Summary string or None if not found.
        """
        self.cursor.execute("""
            SELECT summarize FROM files
            WHERE id = ?
        """, (id,))
        result = self.cursor.fetchone()
        if result:
            logger.info(f"Retrieved summary for file ID: {id}")
            return result
        return None

    def chatlog_table(self, file_id):
        """Generates a unique chatlog table name based on file ID."""
        return "chat_" + hashlib.sha256(file_id.encode()).hexdigest()

    def create_chatlog_table(self, file_id):
        """
        Creates a chatlog table for a given file.

        Args:
            file_id (str): Unique file ID.
        """
        table_name = self.chatlog_table(file_id)
        self.cursor.execute(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role = 'user' OR role = 'assistant'),
                text TEXT NOT NULL
            )
        """)
        self.connect.commit()
        logger.info(f"Created chatlog table: {table_name}")

    def get_history(self, file_id):
        """
        Retrieves the last 100 chat messages for a file.

        Args:
            file_id (str): File ID.

        Returns:
            list: List of chat messages with roles.
        """
        table_name = self.chatlog_table(file_id)
        self.cursor.execute(f"""
            SELECT role, text
            FROM (
                SELECT * FROM {table_name}
                ORDER BY id DESC
                LIMIT 100
            ) query
            ORDER BY id ASC
        """)
        self.connect.commit()
        rows = self.cursor.fetchall()
        return [{"role": role, "content": text} for role, text in rows]

    def log_chat(self, file_id, text, role):
        """
        Logs a new chat message.

        Args:
            file_id (str): File ID.
            text (str): Message content.
            role (str): Either 'user' or 'assistant'.
        """
        table_name = self.chatlog_table(file_id)
        self.cursor.execute(f"""
            INSERT INTO {table_name} (role, text)
            VALUES (?, ?)
        """, (role, text))
        self.connect.commit()
