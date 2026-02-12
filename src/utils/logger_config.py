import contextvars
import logging
import logging.config
import six

logid_context = contextvars.ContextVar("_logid", default=None)

class LogIdFilter(logging.Filter):
    def filter(self, record):
        logid = getattr(record, "tags", {}).get("_logid")
        if logid:
            record._logid = logid
            return True
        logid = logid_context.get() if logid_context.get() else "-"
        record._logid = six.ensure_text(logid)
        return True

class LoggerConfig:
    def __init__(self):
        pass

    def setup_logger(self):
        app_log_file = "exp.log"
        logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": True,
                "formatters": {
                    "default": {
                        "format": "%(asctime)s %(levelname)s %(lineno)d %(processName)s %(_logid)s %(filename)s : %(message)s"
                    }
                },
                "filters": {
                    "logid_filter": {
                        "()": LogIdFilter
                    }
                },
                "handlers": {
                    "console": {
                        "level": "INFO",
                        "class": "logging.StreamHandler",
                        "formatter": "default",
                        "filters": ["logid_filter"],
                    },
                    "file": {
                        "level": "INFO",
                        "class": "logging.handlers.RotatingFileHandler",
                        "filename": app_log_file,
                        "formatter": "default",
                        "filters": ["logid_filter"],
                        "maxBytes": 1024 * 1024 * 50,
                        "backupCount": 20,
                        "encoding": "utf-8",
                    },
                },
                "root": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                },  # ← 这里添加了逗号
                "loggers": {
                    "default": {
                        "handlers": ["console", "file"],
                        "level": "INFO",
                        "propagate": False,
                    },
                },
            }
        )