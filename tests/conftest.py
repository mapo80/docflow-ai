import os, pytest
os.environ.setdefault("API_KEY","test-key")
os.environ.setdefault("MOCK_LLM","1")
os.environ.setdefault("REPORTS_DIR","/mnt/data/reports")
os.environ.setdefault("LOG_LEVEL","DEBUG")
os.environ.setdefault("EMBEDDING_BACKEND","tfidf")

os.environ.setdefault('BACKENDS_MOCK','1')
os.environ.setdefault('TEXT_LAYER_MIN_CHARS','999999')
