import os
import secrets

class Config:
    # Generate a random secret key if none is provided
    SECRET_KEY = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///my_llm_comparator.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # API Keys with proper error handling
    CHATGPT_FREE_API_KEY = os.environ.get('CHATGPT_FREE_API_KEY')
    GEMINI_FREE_API_KEY = os.environ.get('GEMINI_FREE_API_KEY')
    XAI_FREE_API_KEY = os.environ.get('XAI_FREE_API_KEY')
    CLAUDE_FREE_API_KEY = os.environ.get('CLAUDE_FREE_API_KEY')

    # Usage limits and rate limiting
    FREE_TIER_DAILY_LIMIT = int(os.environ.get('FREE_TIER_DAILY_LIMIT', 10))
    RATE_LIMIT_MINUTES = int(os.environ.get('RATE_LIMIT_MINUTES', 1))

    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
