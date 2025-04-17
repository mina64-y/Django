import os
from pathlib import Path
from datetime import timedelta
import dotenv # Import dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
dotenv.load_dotenv(os.path.join(BASE_DIR, '.env'))

SECRET_KEY = os.environ.get('SECRET_KEY', 'fallback-secret-key-only-for-dev') # Load from .env
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't') # Load from .env

ALLOWED_HOSTS = [] # Configure for production

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party apps
    'rest_framework',
    'rest_framework_simplejwt',
    'rest_framework_simplejwt.token_blacklist', # If using blacklist for refresh token revocation
    'corsheaders',
    'django_filters',
    'drf_spectacular',

    # Your apps
    'apps.users',
    'apps.patients',
    'apps.diagnosis',
    'apps.multi_omics',
    # 'apps.medications', # If created as a separate app
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware', # CORS middleware added
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls' # Updated path

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application' # Updated path

# Database (MySQL from .env)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST'),
        'PORT': os.environ.get('DB_PORT'),
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
             'charset': 'utf8mb4', # Recommended for full unicode support
        },
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

# --- Custom User Model ---
AUTH_USER_MODEL = 'users.User' # Point to your custom user model

# Internationalization
LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles' # For production collection

# Media files (User Uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# --- REST Framework Settings ---
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated', # Default to authenticated access
    ),
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
        # Add BrowsableAPIRenderer only if needed during development
        # 'rest_framework.renderers.BrowsableAPIRenderer',
    ),
    'DEFAULT_PARSER_CLASSES': (
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.MultiPartParser',
    ),
    'DEFAULT_FILTER_BACKENDS': (
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.OrderingFilter',
        'rest_framework.filters.SearchFilter',
    ),
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema', # For API docs
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
    'DEFAULT_VERSION': 'v1',
    'ALLOWED_VERSIONS': ['v1'],
    # 'DEFAULT_EXCEPTION_HANDLER': 'config.exceptions.custom_exception_handler', # Optional: for custom error formats
}

# --- JWT Settings ---
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=int(os.environ.get('ACCESS_TOKEN_LIFETIME_MINUTES', 15))), # Short-lived access
    'REFRESH_TOKEN_LIFETIME': timedelta(days=int(os.environ.get('REFRESH_TOKEN_LIFETIME_DAYS', 7))),      # Longer-lived refresh
    'ROTATE_REFRESH_TOKENS': True, # Issue new refresh token when used (good practice)
    'BLACKLIST_AFTER_ROTATION': True, # Invalidate old refresh token after rotation
    'UPDATE_LAST_LOGIN': True,

    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY, # Use Django secret key
    'VERIFYING_KEY': None,
    'AUDIENCE': None,
    'ISSUER': None,
    'JWK_URL': None,
    'LEEWAY': 0,

    'AUTH_HEADER_TYPES': ('Bearer',),
    'AUTH_HEADER_NAME': 'HTTP_AUTHORIZATION',
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
    'USER_AUTHENTICATION_RULE': 'rest_framework_simplejwt.authentication.default_user_authentication_rule',

    'AUTH_TOKEN_CLASSES': ('rest_framework_simplejwt.tokens.AccessToken',),
    'TOKEN_TYPE_CLAIM': 'token_type',
    'TOKEN_USER_CLASS': 'rest_framework_simplejwt.models.TokenUser',

    'JTI_CLAIM': 'jti',

    'SLIDING_TOKEN_REFRESH_EXP_CLAIM': 'refresh_exp',
    'SLIDING_TOKEN_LIFETIME': timedelta(minutes=5), # Inactivity timeout (not used with rotate/blacklist)
    'SLIDING_TOKEN_REFRESH_LIFETIME': timedelta(days=1), # Max lifetime (not used with rotate/blacklist)

    # Consider Whitelisting or Blacklisting for more robust logout/revocation
    # 'BLACKLIST_ENABLED': True, # Requires 'rest_framework_simplejwt.token_blacklist' in INSTALLED_APPS
}

# --- CORS Settings ---
CORS_ALLOWED_ORIGINS = os.environ.get('CORS_ALLOWED_ORIGINS', 'http://localhost:8000').split(',') # Load from .env
# Or be more specific in production:
# CORS_ALLOWED_ORIGINS = [
#     "http://your-react-app.com",
#     "https://your-react-app.com",
# ]
CORS_ALLOW_CREDENTIALS = True # Allow cookies to be sent (needed for JWT HttpOnly refresh token if used)
# CORS_ALLOW_ALL_ORIGINS = False # Set to False for production security

# --- Celery Settings ---
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = TIME_ZONE # Use Django's timezone
CELERY_TASK_TRACK_STARTED = True
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True

# --- Gemini API Key ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# --- Logging Configuration (Essential for PIPA/Debugging) ---
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG' if DEBUG else 'INFO', # Show debug logs only if DEBUG=True
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': { # Example file handler
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': BASE_DIR / 'logs' / 'django.log', # Create 'logs' directory
            'maxBytes': 1024 * 1024 * 5,  # 5 MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'mail_admins': { # Example error email handler
             'level': 'ERROR',
             'class': 'django.utils.log.AdminEmailHandler',
         }
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'], # Add 'mail_admins' for production
            'level': 'INFO',
            'propagate': True,
        },
        'django.request': {
            'handlers': ['mail_admins'], # Email admins on server errors
            'level': 'ERROR',
            'propagate': False,
        },
        # Loggers for your specific apps
        'apps.diagnosis': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG', # Log more details for your app
            'propagate': False,
        },
         'celery': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# --- AI 모델 파일 경로 설정 ---
AI_MODELS_DIR = BASE_DIR / 'ai_models' # 새로 만든 폴더 경로

PANCREAS_CLASSIFY_MODEL_PATH = AI_MODELS_DIR / 'best_model_auc_20250408-125244.pth' 
PANCREAS_SEGMENT_MODEL_PATH = AI_MODELS_DIR / 'best_model_20250408-122756.pth'    
GENE_MODEL_PATH = AI_MODELS_DIR / 'best_random_forest_model.pkl'
PROTEIN_MODEL_PATH = AI_MODELS_DIR / 'pro_MLPClassifier_best_model.pkl'
METHYLATION_MODEL_PATH = AI_MODELS_DIR / 'methyl_SVM_best_model.pkl'
CNV_MODEL_PATH = AI_MODELS_DIR / 'cnv-k-NN_best_model.pkl'

# --- Multi-omics Columns Files ---
GENE_COLUMNS_PATH = AI_MODELS_DIR / 'gene_columns.pkl'
PROTEIN_COLUMNS_PATH = AI_MODELS_DIR / 'protein_columns.pkl'
METHYLATION_COLUMNS_PATH = AI_MODELS_DIR / 'methylation_columns.pkl'
CNV_COLUMNS_PATH = AI_MODELS_DIR / 'cnv_columns.pkl'
    
# --- drf-spectacular Settings (API Docs) ---
SPECTACULAR_SETTINGS = {
    'TITLE': 'Medical CDSS API',
    'DESCRIPTION': 'API for Pancreas Cancer Diagnosis and Medication Management',
    'VERSION': '1.0.0',
    'SERVE_INCLUDE_SCHEMA': False, # Don't serve schema.yml directly
    # OTHER SETTINGS
}

# Security Settings (Review and uncomment/configure for production)
# CSRF_COOKIE_SECURE = True # Use True in production over HTTPS
# SESSION_COOKIE_SECURE = True # Use True in production over HTTPS
# SECURE_SSL_REDIRECT = True # Use True in production over HTTPS
# SECURE_HSTS_SECONDS = 31536000 # Example: 1 year
# SECURE_HSTS_INCLUDE_SUBDOMAINS = True
# SECURE_HSTS_PRELOAD = True
# SECURE_CONTENT_TYPE_NOSNIFF = True
# SECURE_BROWSER_XSS_FILTER = True
# X_FRAME_OPTIONS = 'DENY'