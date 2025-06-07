from flask import (Flask, request, jsonify,render_template,redirect,url_for,session,send_file,Response,make_response)
from pathlib import Path
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
import json
import threading
import time
import fitz
import base64
import logging
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import requests
from bs4 import BeautifulSoup
import re
import html
import backoff
import urllib.parse
import itertools
from time import sleep
from random import uniform
from requests.exceptions import RequestException
from duckduckgo_search import DDGS
from urllib.parse import urlparse

# Document Processing imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, OperationalError

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type"],
    "supports_credentials": True
}})


load_dotenv()


class Config:
    UPLOAD_FOLDER = Path('uploads')
    TEMP_FOLDER = Path('temp')
    ALLOWED_EXTENSIONS = {'pdf', 'docx'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    SESSION_TIMEOUT = 3600
    CLEANUP_INTERVAL = 300

    # Database Configuration
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = urllib.parse.quote_plus(os.getenv('DB_PASSWORD', 'STPL123'))
    DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
    DB_PORT = os.getenv('DB_PORT', '3306')
    DB_NAME = os.getenv('DB_NAME', 'document_chat')

    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {

        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True
    }

app.config.from_object(Config)
app.secret_key = os.urandom(24)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    documents = db.relationship('DocumentModel', backref='user', lazy=True)
    chat_history = db.relationship('ChatHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class DocumentModel(db.Model):
    __tablename__ = 'documents'

    id = db.Column(db.String(36), primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    chroma_collection = db.Column(db.String(100), nullable=False)
    file_size = db.Column(db.Integer, nullable=True)
    mime_type = db.Column(db.String(100), nullable=True)

    __table_args__ = (
        db.Index('idx_session_upload', session_id, upload_date),
    )

class ChatHistory(db.Model):
    __tablename__ = 'chat_history'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    document_id = db.Column(db.String(36), db.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    document = db.relationship('DocumentModel', backref=db.backref('chat_history', lazy=True))

    __table_args__ = (
        db.Index('idx_document_timestamp', document_id, timestamp),
        db.Index('idx_session_document', session_id, document_id),
    )

class WebSearchHistory(db.Model):
    __tablename__ = 'web_search_history'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    document_id = db.Column(db.String(36), db.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(36), nullable=False, index=True)
    search_query = db.Column(db.Text, nullable=False)
    search_results = db.Column(db.Text, nullable=True)  # You can store search result summaries or URLs
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    document = db.relationship('DocumentModel', backref=db.backref('web_search_history', lazy=True))
    user = db.relationship('User', backref=db.backref('web_search_history', lazy=True))

    __table_args__ = (
        db.Index('idx_document_timestamp_search', document_id, timestamp),
        db.Index('idx_session_search', session_id, document_id),
    )

# Database initialization function
def init_tables():
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            print(" Database tables created successfully!")
        except Exception as e:
            print(f" Error creating tables: {e}")
            raise

def init_db():
    try:
        # Create database engine without database name first
        base_engine = create_engine(f"mysql+pymysql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/")

        # Try to create database if it doesn't exist
        with base_engine.connect() as conn:
            # Drop the existing database to ensure clean slate
            conn.execute(text("DROP DATABASE IF EXISTS document_chat"))
            conn.commit()

            # Create the database
            conn.execute(text("CREATE DATABASE document_chat"))
            conn.commit()

            # Use the database
            conn.execute(text("USE document_chat"))
            conn.commit()

        # Create all tables with the new schema
        with app.app_context():
            db.create_all()

        print(" Database initialized successfully!")

    except Exception as e:
        print(f"❌ Error during database initialization: {e}")
        raise

# Add this to ensure database connection is working
def verify_db_connection():
    try:
        with app.app_context():
            # Try to query the documents table
            DocumentModel.query.first()
            print(" Database connection verified!")
            return True
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False


class RateLimitedSearcher:
    def __init__(self, max_retries=3, base_delay=2):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = requests.Session()
        self.ddgs = DDGS()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1'
        })

    @backoff.on_exception(backoff.expo, RequestException, max_tries=3)
    def _fetch_page_description(self, url):
        try:
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                return meta_desc['content']

            first_p = soup.find('p')
            if first_p:
                return first_p.get_text().strip()

            return None
        except Exception as e:
            logger.warning(f"Error fetching page description: {str(e)}")
            return None

    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = html.unescape(text)
        return text.strip()[:200]

    def search(self, query):
        results = []
        try:
            search_generator = self.ddgs.text(
                keywords=query,
                region='wt-wt',
                safesearch='moderate'
            )

            raw_results = list(itertools.islice(search_generator, 5))

            for result in raw_results:
                try:
                    url = result.get('href') or result.get('url')
                    if not url:
                        continue

                    title = self.clean_text(result.get('title', ''))
                    if not title:
                        parsed_url = urlparse(url)
                        title = parsed_url.netloc

                    description = self.clean_text(result.get('body', '') or result.get('snippet', ''))
                    if not description:
                        description = self.clean_text(self._fetch_page_description(url))
                    if not description:
                        description = f"Web page from {urlparse(url).netloc}"

                    results.append({
                        "url": url,
                        "title": title,
                        "summary": description
                    })

                    sleep(uniform(0.5, 1.5))

                except Exception as e:
                    logger.warning(f"Error processing search result: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Search error: {str(e)}")

        return results

class DocumentProcessor:
    def __init__(self, session_id, user_id):
        self.session_id = session_id
        self.user_id = user_id
        self.embedding_model = HuggingFaceEmbeddings()
        self.conversation_chain = None
        self.vectorstore = None
        self.current_document_id = None  # Explicitly set to None
        self.chat_history = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.is_document_loaded = False  # Explicitly set to False
        self.last_access = time.time()
        self.current_pdf_path = None
        self.page_mapping = {}
        self.doc_pages = None
        self.persist_directory = Path("chroma_db")
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    def _initialize_llm(self):
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
            groq_api_key=Config.GROQ_API_KEY
        )

    def _get_text_coordinates(self, page_num, text):
        if not self.doc_pages or page_num >= len(self.doc_pages):
            return []

        page = self.doc_pages[page_num]
        text_instances = page.search_for(text)

        coordinates = []
        for inst in text_instances:
            coordinates.append({
                'x': inst.x0,
                'y': inst.y0,
                'width': inst.x1 - inst.x0,
                'height': inst.y1 - inst.y0
            })

        return coordinates

    def process_document(self, file_path, original_filename):
        try:
            if not self.user_id:
                raise ValueError("User must be authenticated to upload documents")

            document_id = str(uuid4())
            collection_name = f"collection_{document_id}"

            file_extension = file_path.suffix.lower()
            if file_extension == '.pdf':
                self.current_pdf_path = file_path
                doc = fitz.open(str(file_path))
                self.doc_pages = [page for page in doc]
                documents = []
                for page_num, page in enumerate(self.doc_pages):
                    text = page.get_text()
                    doc = Document(
                        page_content=text,
                        metadata={"page": page_num + 1}
                    )
                    documents.append(doc)
                    self.page_mapping[text] = page_num + 1
            else:
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load()

            with app.app_context():
                db_document = DocumentModel(
                    id=document_id,
                    filename=original_filename,
                    filepath=str(file_path),
                    session_id=self.session_id,
                    user_id=self.user_id,
                    chroma_collection=collection_name
                )
                db.session.add(db_document)
                db.session.commit()

            self._setup_conversation_chain(documents, collection_name)
            processed_data = self._process_documents(documents)

            self.current_document_id = document_id
            self.is_document_loaded = True
            self.last_access = time.time()

            if file_extension == '.pdf':
                with open(file_path, "rb") as f:
                    processed_data['pdf_base64'] = base64.b64encode(f.read()).decode()

            return processed_data

        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            raise

    def _process_documents(self, documents):
        try:
            page_summaries = {}
            full_text = []

            for i, doc in enumerate(documents):
                full_text.append(doc.page_content)
                summary = self._create_summary(doc.page_content)
                if summary:
                    page_summaries[str(i + 1)] = summary

            full_summary = self._create_summary(" ".join(full_text)) if full_text else ""

            return {
                "full_summary": full_summary,
                "page_summaries": page_summaries,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _create_summary(self, text):
        try:
            sentences = text.split('.')
            summary_sentences = sentences[:3]
            summary = '. '.join(sentence.strip() for sentence in summary_sentences if sentence.strip())
            return summary + '.' if summary else ""
        except Exception as e:
            logger.error(f"Summary creation error: {str(e)}")
            return ""

    def _setup_conversation_chain(self, documents, collection_name):
        try:
            texts = self.text_splitter.split_documents(documents)

            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding_model,
                persist_directory=str(self.persist_directory),
                collection_name=collection_name
            )

            self.vectorstore.persist()

            llm = self._initialize_llm()
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                verbose=True
            )

        except Exception as e:
            logger.error(f"Conversation chain setup error: {str(e)}")
            raise

    def load_existing_document(self, document_id):
        try:
            with app.app_context():
                document = DocumentModel.query.get(document_id)
                if not document:
                    raise ValueError("Document not found")

                file_path = Path(document.filepath)
                if not file_path.exists():
                    raise ValueError("Document file not found")

                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_directory),
                    embedding_function=self.embedding_model,
                    collection_name=document.chroma_collection
                )

                llm = self._initialize_llm()
                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    verbose=True
                )

                self.chat_history = [
                    (ch.question, ch.answer)
                    for ch in ChatHistory.query.filter_by(
                        document_id=document_id,
                        session_id=self.session_id
                    ).order_by(ChatHistory.timestamp)
                ]

                self.current_document_id = document_id
                self.is_document_loaded = True
                self.last_access = time.time()

                if file_path.suffix.lower() == '.pdf':
                    self.current_pdf_path = file_path
                    doc = fitz.open(str(file_path))
                    self.doc_pages = [page for page in doc]

                return {"status": "success", "message": "Document loaded successfully"}

        except Exception as e:
            logger.error(f"Error loading existing document: {str(e)}")
            raise

    def highlight_pdf(self, sources):
        if not self.current_pdf_path or not self.current_pdf_path.exists():
            logger.error("No PDF document loaded")
            return None

        temp_doc = None
        temp_path = None
        try:
            temp_doc = fitz.open(str(self.current_pdf_path))
            temp_path = Config.TEMP_FOLDER / f"{time.time_ns()}_highlighted.pdf"

            for source in sources:
                page_num = source.get('page', 1) - 1
                text = source.get('text', '')
                if 0 <= page_num < temp_doc.page_count:
                    page = temp_doc[page_num]
                    text_instances = page.search_for(text)
                    for inst in text_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()

            temp_doc.save(str(temp_path))

            with open(temp_path, "rb") as f:
                encoded_pdf = base64.b64encode(f.read()).decode()

            return encoded_pdf

        except Exception as e:
            logger.error(f"Error highlighting PDF: {str(e)}")
            return None

        finally:
            if temp_doc:
                temp_doc.close()
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def ask_question(self, question):
        try:
            if not self.is_document_loaded or not self.conversation_chain:
                return {"error": "Please upload a document first", "status": "error"}

            if not self.user_id:
                return {"error": "User must be authenticated", "status": "error"}

            self.last_access = time.time()

            response = self.conversation_chain({
                "question": question,
                "chat_history": self.chat_history
            })

            with app.app_context():
                chat_entry = ChatHistory(
                    document_id=self.current_document_id,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    question=question,
                    answer=response['answer']
                )
                db.session.add(chat_entry)
                db.session.commit()

            self.chat_history.append((question, response['answer']))

            sources = []
            if 'source_documents' in response and response['source_documents']:
                for doc in response['source_documents']:
                    source_text = doc.page_content
                    page_num = doc.metadata.get('page', 1)
                    coordinates = self._get_text_coordinates(page_num - 1, source_text) if hasattr(self, '_get_text_coordinates') else []
                    source = {
                        'page': page_num,
                        'text': source_text,
                        'coordinates': coordinates
                    }
                    sources.append(source)

            highlighted_pdf = self.highlight_pdf(sources) if hasattr(self, 'highlight_pdf') and sources else None

            return {
                "answer": response['answer'],
                "sources": sources,
                "highlighted_pdf": highlighted_pdf,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Question processing error: {str(e)}", exc_info=True)
            return {"status": "error", "error": str(e)}



    def web_search(self, query):
        """
        Perform web search using RateLimitedSearcher
        """
        try:
            # Check if document is loaded before proceeding
            if not self.is_document_loaded or not self.current_document_id:
                raise ValueError("No document loaded. Please load a document before performing web search.")

            self.last_access = time.time()
            searcher = RateLimitedSearcher()
            return searcher.search(query)
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            raise

# Session management
processors = {}
processor_lock = threading.Lock()

def get_processor():
    if 'session_id' not in session:
        session['session_id'] = str(uuid4())

    session_id = session['session_id']
    user_id = current_user.id if current_user.is_authenticated else None

    with processor_lock:
        if session_id not in processors:
            processors[session_id] = DocumentProcessor(session_id, user_id)
        processors[session_id].last_access = time.time()
        return processors[session_id]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        try:
            data = request.get_json()

            # Validate required fields
            if not all(key in data for key in ['username', 'email', 'password']):
                return jsonify({'error': 'Missing required fields'}), 400

            # Check if username already exists
            if User.query.filter_by(username=data['username']).first():
                return jsonify({'error': 'Username already exists'}), 400

            # Check if email already exists
            if User.query.filter_by(email=data['email']).first():
                return jsonify({'error': 'Email already exists'}), 400

            # Create new user
            new_user = User(
                username=data['username'],
                email=data['email']
            )
            new_user.set_password(data['password'])

            # Add to database
            db.session.add(new_user)
            db.session.commit()

            return jsonify({"message": "Registration successful"})

        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration error: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            data = request.get_json()
            logger.debug(f"Login attempt for username: {data.get('username', 'not provided')}")

            if not data:
                logger.error("No data received in login request")
                return jsonify({'error': 'No data provided'}), 400

            if not all(key in data for key in ['username', 'password']):
                logger.error("Missing required login fields")
                return jsonify({'error': 'Username and password are required'}), 400

            username = data['username']
            password = data['password']

            if not username or not password:
                logger.error("Empty username or password")
                return jsonify({'error': 'Username and password cannot be empty'}), 400

            user = User.query.filter_by(username=username).first()
            logger.debug(f"User found: {user is not None}")

            if user and user.check_password(password):
                login_user(user)
                session['user_id'] = user.id
                session['username'] = user.username
                logger.info(f"Successful login for user: {username}")

                response = make_response(jsonify({
                    'message': 'Logged in successfully',
                    'user': {
                        'username': user.username,
                        'email': user.email
                    },
                    'redirect': url_for('index')
                }))
                return response
            else:
                logger.warning(f"Failed login attempt for username: {username}")
                return jsonify({'error': 'Invalid username or password'}), 401

        except Exception as e:
            logger.error(f"Login error: {str(e)}", exc_info=True)
            return jsonify({'error': 'An error occurred during login'}), 500

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'})

# Document management routes
@app.route('/upload', methods=['POST', 'OPTIONS'])
@login_required
def upload_file():
    if request.method == 'OPTIONS':
        return app.make_default_options_response()

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        filename = secure_filename(file.filename)
        file_path = Config.UPLOAD_FOLDER / filename
        file.save(str(file_path))

        processor = get_processor()
        result = processor.process_document(file_path, filename)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def create_templates():
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)

    # Write templates if they don't exist
    if not (templates_dir / 'login.html').exists():
        (templates_dir / 'login.html').write_text(login_template)

    if not (templates_dir / 'register.html').exists():
        (templates_dir / 'register.html').write_text(register_template)

    if not (templates_dir / 'index.html').exists():
        (templates_dir / 'index.html').write_text(index_template)

    if not (templates_dir / 'chat_history.html').exists():
        (templates_dir / 'chat_history.html').write_text(chat_history_template)

    document = {
        "document_name": "Sample Document",
        "document_url": "https://your-storage-service-link/sample-document.pdf"
    }

    # Ensure Flask app context is set up
    with app.app_context():
        return render_template("chat_history.html", document=document)

# Call the function
create_templates()



# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return redirect(url_for('login'))

@app.route('/index')
@login_required
def index():
    return render_template('index.html', username=current_user.username)


@app.route('/documents', methods=['GET'])
def get_documents():
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session found'}), 401

        documents = DocumentModel.query.filter_by(session_id=session_id).all()
        return jsonify({
            'documents': [{
                'id': doc.id,
                'filename': doc.filename,
                'upload_date': doc.upload_date.isoformat()
            } for doc in documents]
        })

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat-history', methods=['GET'])
@login_required
def view_chat_history():
    try:
        # Get all documents for the current user
        documents = DocumentModel.query.filter_by(user_id=current_user.id).all()

        # Get chat history for all documents
        history_data = []
        for doc in documents:
            chat_history = ChatHistory.query.filter_by(
                document_id=doc.id,
                user_id=current_user.id
            ).order_by(ChatHistory.timestamp.desc()).all()

            if chat_history:
                history_data.append({
                    'document_name': doc.filename,
                    'document_id': doc.id,
                    'history': [{
                        'question': chat.question,
                        'answer': chat.answer,
                        'timestamp': chat.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    } for chat in chat_history]
                })

        return render_template('chat_history.html', history_data=history_data)

    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat-history/<document_id>', methods=['GET'])
@login_required
def get_document_chat_history(document_id):
    try:
        # Verify the document belongs to the current user
        document = DocumentModel.query.filter_by(
            id=document_id,
            user_id=current_user.id
        ).first()

        if not document:
            return jsonify({'error': 'Document not found'}), 404

        chat_history = ChatHistory.query.filter_by(
            document_id=document_id,
            user_id=current_user.id
        ).order_by(ChatHistory.timestamp.desc()).all()

        history_data = [{
            'question': chat.question,
            'answer': chat.answer,
            'timestamp': chat.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        } for chat in chat_history]

        return jsonify({'history': history_data})

    except Exception as e:
        logger.error(f"Error retrieving document chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search-history', methods=['GET'])
@login_required
def get_search_history():
    try:
        # Get search history for the current user
        history = WebSearchHistory.query.filter_by(
            user_id=current_user.id
        ).order_by(WebSearchHistory.timestamp.desc()).all()

        history_data = [{
            'search_query': item.search_query,
            'search_results': json.loads(item.search_results) if item.search_results else None,
            'timestamp': item.timestamp.isoformat(),
            'document_id': item.document_id
        } for item in history]

        return jsonify({'history': history_data})

    except Exception as e:
        logger.error(f"Error retrieving search history: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/documents/<document_id>/load', methods=['POST'])
def load_document(document_id):
    try:
        processor = get_processor()
        result = processor.load_existing_document(document_id)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        return jsonify({'error': str(e)}), 500


def generate_response(answer):
    """Generator function to stream the response"""
    if isinstance(answer, str):
        yield answer
    else:
        yield json.dumps(answer)
@app.route('/ask', methods=['POST'])
@login_required
def ask():
    try:
        data = request.json
        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        processor = get_processor()
        processor.user_id = current_user.id  # Ensure user_id is set
        response = processor.ask_question(question)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during question handling: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/search', methods=['POST'])
@login_required
def web_search_route():
    try:
        data = request.json
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'results': [], 'error': 'No query provided'}), 400

        processor = get_processor()

        # Check if a document is loaded
        if not processor.is_document_loaded or not processor.current_document_id:
            return jsonify({
                'results': [],
                'error': 'Please load a document before performing web search'
            }), 400

        results = processor.web_search(query)

        try:
            # Store search history only if we have a valid document_id
            search_history = WebSearchHistory(
                document_id=processor.current_document_id,  # This should now be valid
                user_id=current_user.id,
                session_id=processor.session_id,
                search_query=query,
                search_results=json.dumps(results)
            )
            db.session.add(search_history)
            db.session.commit()
        except Exception as db_error:
            logger.warning(f"Failed to save search history: {str(db_error)}")
            # Don't fail the whole request if just the history save fails
            db.session.rollback()

        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        db.session.rollback()
        return jsonify({'results': [], 'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = Config.UPLOAD_FOLDER / secure_filename(filename)
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True)
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup_old_processors():
    while True:
        try:
            current_time = time.time()
            with processor_lock:
                for session_id, processor in list(processors.items()):
                    if current_time - processor.last_access > Config.SESSION_TIMEOUT:
                        del processors[session_id]
            time.sleep(Config.CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error during processor cleanup: {str(e)}")
            time.sleep(60)

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_processors, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    try:
        # Initialize database and tables
        init_db()

        # Verify database connection
        if not verify_db_connection():
            raise Exception("Database verification failed")

        # Ensure the upload and temp directories exist
        Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        Config.TEMP_FOLDER.mkdir(parents=True, exist_ok=True)

        # Start the Flask application
        app.run(debug=True, port=8080)

    except Exception as e:
        print(f" Application startup error: {e}")