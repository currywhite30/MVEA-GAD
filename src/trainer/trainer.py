from tqdm import tqdm
from src.model.model_loader import ModelLoader
from src.data.data_preprocess import DataPreprocessor
from src.eval.evaluate import Evaluator
import torch
import logging

logger = logging.getLogger()

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model_loader = ModelLoader(self.config)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.evaluator = Evaluator()

    def train(self,):
        auc_list, ap_list, recall_list = [], [], []
        for _ in tqdm(range(self.config.num_trail)):
            detector = self.model_loader.get(trail=_)
            data = self.data_preprocessor.get()
            data = self.data_preprocessor.preprocess(data)
            if self.config.model.name == 'if' or self.config.model.name == 'lof':
                test_result = detector.fit(data.x, label=data.y.bool())
            else:
                test_result = detector.fit(data=data, label=data.y.bool())
            if test_result is not None:
                auc, rec, pre, ap, f1 = test_result
                auc_list.append(auc)
                ap_list.append(ap)
                recall_list.append(rec)
        auc_tensor = torch.tensor(auc_list)
        ap_tensor = torch.tensor(ap_list)
        recall_tensor = torch.tensor(recall_list)
        logger.info(">>>src.trainer.trainer.py|>>>Trainer.train()|>>>dataset: {} model: {} AUC: {:.4f}±{:.4f} ({:.4f})\nAP: {:.4f}±{:.4f} ({:.4f})\nRecall: {:.4f}±{:.4f} ({:.4f})".format(self.config.dataset.name, self.config.model.name, torch.mean(auc_tensor), torch.std(auc_tensor), torch.max(auc_tensor), torch.mean(ap_tensor), torch.std(ap_tensor), torch.max(ap_tensor), torch.mean(recall_tensor), torch.std(recall_tensor), torch.max(recall_tensor)))
        return torch.mean(auc_tensor)