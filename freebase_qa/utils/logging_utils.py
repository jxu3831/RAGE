import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_file: str = 'freebase_qa.log', level=logging.INFO):
    """配置日志系统"""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )

    # 降低第三方 HTTP 库的日志等级，防止输出冗余请求信息
    for noisy_logger in ['httpx', 'urllib3', 'dashscope']:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)