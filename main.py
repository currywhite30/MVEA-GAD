import hydra
from src.trainer.trainer import Trainer
from src.utils.logger_config import LoggerConfig
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)

@hydra.main(config_path="/data/baisongran/evidential-graph-anomaly/conf", config_name="config", version_base=None)
def main(config):
    # LoggerConfig().setup_logger()
    logger = logging.getLogger()
    logger.info("Starting the application...")
    trainer = Trainer(config)
    auc_mean = trainer.train()
    logger.info("Application finished successfully.")
    # === 新增这一行 ===
    # 这样脚本就能通过 grep "Final Result AUC" 抓取到具体数值
    logger.info(f"Final Result AUC: {auc_mean}") 
    # ==================
    return auc_mean
if __name__ == "__main__":
    main()