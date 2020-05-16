import logging.config
import os
from src.comman import Constant



LOG_PATH = Constant.LOG_PATH
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            # 'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'null': {  # a NullHandler, which will pass any DEBUG (or higher) message to /dev/null.
            'level': 'DEBUG',
            'class': 'logging.NullHandler',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'src':
            {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                "filename": os.path.join(LOG_PATH, 'src.log'),
                'formatter': 'verbose'
            },

    },
    'loggers': {
        'src': {
            'handlers': ['src', 'console'],
            'level': 'DEBUG',
        }
    }
}


class Logger:
    def __init__(self):
        logging.config.dictConfig(LOGGING)
        self.__logger = logging.getLogger(__name__)

    @property
    def logger(self):
        return self.__logger
