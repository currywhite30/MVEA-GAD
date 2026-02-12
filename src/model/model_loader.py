from src.model.hega import DOMINANTHega
import logging

logger = logging.getLogger()

class ModelLoader:
    def __init__(self, config):
        self.config = config

    def get(self, trail):
        model_name = self.config.model.name
        if model_name == "hega":
            return DOMINANTHega(
                            backbone=self.config.model.backbone,
                            backbone_norm=self.config.model.backbone_norm,
                            hid_dim=self.config.model.hid_dim,
                            z_dim=self.config.model.z_dim,
                            num_layers=self.config.model.num_layers,
                            n_clusters=self.config.model.n_clusters,
                            sigmoid_s=self.config.model.sigmoid_s,
                            weight_decay=self.config.optimizer.weight_decay,
                            dropout=self.config.optimizer.dropout,
                            lr=self.config.optimizer.lr,
                            epoch=self.config.epoch,
                            patience=self.config.patience,
                            gpu=self.config.gpu,
                            weight=self.config.model.alpha,
                            batch_size=self.config.batch_size,
                            num_neigh=self.config.model.num_neigh,
                            verbose=self.config.verbose,
                            trail=trail,
                            attr_loss_weight=self.config.model.attr_loss_weight,
                            has_annealing=self.config.model.has_annealing,
                            attr_reg_base=self.config.model.attr_reg_base,
                            attr_warmup_ratios=self.config.model.attr_warmup_ratios,
                            clustering_weight=self.config.model.clustering_weight,
                            struct_loss_weight=self.config.model.struct_loss_weight,
                            bce_weight=self.config.model.bce_weight,
                            ranking_weight=self.config.model.ranking_weight,
                            dice_weight=self.config.model.dice_weight,
                            misleading_weight=self.config.model.misleading_weight,
                            pos_loss_weight=self.config.model.pos_loss_weight,
                            neg_loss_weight=self.config.model.neg_loss_weight,
                            kl_weight=self.config.model.kl_weight,
                            ranking_margin=self.config.model.ranking_margin,
                            neighbor_constrain_weight=self.config.model.neighbor_constrain_weight,
                            lambda_attr_recon=self.config.model.lambda_attr_recon,
                            lambda_clustering_recon=self.config.model.lambda_clustering_recon,
                            lambda_struct_recon=self.config.model.lambda_struct_recon,
                            lambda_neighbor_recon=self.config.model.lambda_neighbor_recon,
                            lambda_attr_unc=self.config.model.lambda_attr_unc,
                            lambda_clustering_unc=self.config.model.lambda_clustering_unc,
                            lambda_struct_unc=self.config.model.lambda_struct_unc,
                            lambda_recon=self.config.model.lambda_recon,
                            lambda_unc=self.config.model.lambda_unc,
                            anomaly_func_type=self.config.model.anomaly_func_type,
                        )