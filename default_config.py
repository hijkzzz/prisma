CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

MAIL_SERVER = 'smtp.qq.com'
MAIL_PROT = 465
MAIL_USE_TLS = True
MAIL_USE_SSL = False
MAIL_USERNAME = ''
MAIL_PASSWORD = ''
MAIL_DEBUG = True

UPLOAD_FOLDER = 'upload/'
ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.jpeg'])
MAX_CONTENT_LENGTH = 4 * 1024 * 1024

MODEL_FOLDER = 'models/'
MODEL_FILES = set(['crayon.ckpt', 'feathers.ckpt', 'sketch.ckpt',
                   'starry.ckpt', 'ink.ckpt', 'mosaic.ckpt', 'wave.ckpt'])
OUTPUT_FOLDER = 'generate/'
