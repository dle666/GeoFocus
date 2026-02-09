import logging
import os
from datetime import datetime

def setup_logger():
    os.makedirs("logs", exist_ok=True)
    
    # 添加当前时间到文件名，例如 logs/training_2025-05-02_15-30-00.log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/code_rewards_{timestamp}.log"

    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logger()
