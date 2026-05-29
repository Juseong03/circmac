from typing import Any, Dict, List, Tuple, Optional

import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.model import ModelWrapper
from transformers import T5EncoderModel
from multimolecule.models import RnaBertModel, RnaErnieModel, RnaFmModel, RnaMsmModel
from utils import seed_everything, get_optimizer, save_model, load_model, save_logs
from utils import create_ntp_inputs_and_labels_with_mask
from utils import create_mlm_inputs_and_labels, calculate_class_weights_from_df
from utils import cal_loss_bind, cal_loss_sites, cal_score_bind, cal_score_sites
from utils import cal_score_self, UncertaintyWeightingLoss
from utils import focal_loss, cal_loss_sites_focal, batched_focal_loss

# --- Trainer Class ---
class Trainer:
    def __init__(
        self,
        seed: int = 42,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        dir_save: str = './saved_models/',
        dir_log: str = './logs/',
        experiment_name: str = 'experiment',
        verbose: bool = True
    ) -> None:
        # Set the seed and device, and create save and log directories.
        self.seed = seed
        seed_everything(seed)
        self.device = torch.device(device)

        self.experiment_name = experiment_name
        self.dir_save = dir_save
        self.dir_log = dir_log

        os.makedirs(self.dir_save, exist_ok=True)
        os.makedirs(self.dir_log, exist_ok=True)
        self.verbose = verbose

        # Initialize loss function, best scores, patience, and log containers.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.best_score = 0.0
        self.best_epoch = 0
        self.patience = 0
        self.patience_max = 20

        self.alpha = None  # For uncertainty weighting
        self.beta = None   # For uncertainty weighting
        self.logs = {'train': {}, 'valid': {}, 'test': {}, 'extra': {}, 'final': {}}
        
        self.model_name = "default_model"
        self.setup_logging("init")

        self.extra_loader = None  # Placeholder for any extra dataset if needed.
        self.use_unified_head = False  # Default to legacy separate heads approach

    def setup_logging(self, log_name: str = 'training') -> None:
        self.log_dir = os.path.join(self.dir_log, self.model_name, self.experiment_name, str(self.seed))
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_file = os.path.join(self.log_dir, f'{log_name}.log')

        self.logger = logging.getLogger(f"{self.model_name}_{self.experiment_name}_{self.seed}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(file_handler)

            if self.verbose:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(console_handler)

        self.logger.propagate = False  # Prevent double logging
        self.logger.info("✅ Logging setup complete.")

    def define_model(
        self,
        config,
        model_name: str = 'mamba',
        pretrain: bool = False,
        is_convblock: bool = False,
        is_cross_attention: bool = False,
        interaction: str = 'concat',
        pooling_mode_target: str = 'cls',
        binding_pooling: str = 'mean',
        site_head_type: str = 'conv1d'
    ) -> None:
        self.model_name = model_name
        self.pretrain_mode = pretrain
        self.use_unified_head = True # Proposed: site-first approach
        self.binding_pooling = binding_pooling
        self.interaction = interaction
        self.model = ModelWrapper(
            config=config,
            name=model_name,
            device=self.device,
            pooling_mode_target=pooling_mode_target
        ).to(self.device)
        self.model._init_embedding()
        self.model._get_backbone()
        if is_convblock:
            self.model._set_convblock()
        if is_cross_attention:
            self.model._set_cross_attention()

        # Site-first approach: sites prediction is main task, binding derived from sites
        self.model._set_unified_site_head(binding_pooling=binding_pooling, interaction=interaction)

        if pretrain:
            self.model._set_token_dropout()
            self.model._set_mlm_head()
            self.model._set_ssp_head()
            # Use CircularPairingHead for circmac model
            use_circular_pairing = model_name.lower() == 'circmac'
            self.model._set_pairing_head(circular=use_circular_pairing)

        self.logger.info(f"Model '{model_name}' initialized. Pretraining mode: {pretrain}, Interaction: {interaction}")

    def define_pretrained_model(self, model_name: str = 'rnabert') -> None:
        if model_name.lower() == 'rnabert':
            self.model_pt = RnaBertModel.from_pretrained('multimolecule/rnabert').to(self.device)
        elif model_name.lower() == 'rnaernie':
            self.model_pt = RnaErnieModel.from_pretrained('multimolecule/rnaernie').to(self.device)
        elif model_name.lower() == 'rnafm':
            self.model_pt = RnaFmModel.from_pretrained('multimolecule/rnafm').to(self.device)
        elif model_name.lower() == 'rnamsm':
            self.model_pt = RnaMsmModel.from_pretrained('multimolecule/rnamsm').to(self.device)
        else:
            raise ValueError(f"Pretrained model '{model_name}' not recognized.")
        self.model_pt.eval()
        for param in self.model_pt.parameters():
            param.requires_grad = False

    def set_pretrained_target(self, target: str = 'mirna', rna_model: str = 'rnabert') -> None:
        if target.lower() in ['mirna', 'mirnas', 'micro']:
            if rna_model.lower() == 'rnabert':
                self.model_target = RnaBertModel.from_pretrained('multimolecule/rnabert').to(self.device)
            elif rna_model.lower() == 'rnaernie':
                self.model_target = RnaErnieModel.from_pretrained('multimolecule/rnaernie').to(self.device)
            elif rna_model.lower() == 'rnafm':
                self.model_target = RnaFmModel.from_pretrained('multimolecule/rnafm').to(self.device)
            elif rna_model.lower() == 'rnamsm':
                self.model_target = RnaMsmModel.from_pretrained('multimolecule/rnamsm').to(self.device)
            else:
                raise ValueError(f"Target model '{rna_model}' not recognized for mirna.")
            d_target = self.model_target.embeddings.word_embeddings.weight.data.shape[1]
        elif target.lower() in ['protein', 'rbp']:
            self.model_target = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(self.device)
            d_target = self.model_target.shared.weight.shape[1]
        else:
            raise ValueError(f"Target '{target}' not recognized.")
        self.model_target.eval()
        for param in self.model_target.parameters():
            param.requires_grad = False
        # Set the target projection layer in the main model.
        self.model._set_proj_target(d_target)
        if self.verbose:
            print(f"Target model for {target} set with projection dimension {d_target}")
            self.logger.info(f"Target model for {target} set with projection dimension {d_target}")

    def set_dataloader(self, dataset, batch_size: int = 32, part: int = 0, shuffle: bool = True, num_workers: int = 0) -> None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True if num_workers > 0 else False)
        if part == 0:
            self.train_loader = dataloader
        elif part == 1:
            self.valid_loader = dataloader
        elif part == 2:
            self.test_loader = dataloader
        elif part == 3:
            self.extra_loader = dataloader
        else:
            raise ValueError("Part must be 0 (train), 1 (valid) or 2 (test).")

    def set_dataloader_self(self, dataset, batch_size: int = 32, part: int = 0, shuffle: bool = True, num_workers: int = 0) -> None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True if num_workers > 0 else False)

        if part == 0:
            self.train_self = dataloader
        elif part == 1:
            self.valid_self = dataloader
        elif part == 2:
            self.test_self = dataloader
        else:
            raise ValueError('Invalid part, must be 0 (train), 1 (valid), or 2 (test).')

    def set_optimizer(self, optimizer_name: str = 'adam', lr: float = 1e-4, w_decay: float = None, freeze: bool = False) -> None:
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            # Proposed: Only train unified_site_head
            trainable_params = list(self.model.unified_site_head.parameters())

            for param in trainable_params:
                param.requires_grad = True
            self.optimizer = get_optimizer(trainable_params, optimizer_name, lr=lr, w_decay=w_decay)
            if self.verbose:
                print(f"Optimizer set with frozen backbone; trainable parameters: {len(trainable_params)}")
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            self.optimizer = get_optimizer(self.model.parameters(), optimizer_name, lr=lr, w_decay=w_decay)
            if self.verbose:
                total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"Optimizer set for full model training; trainable parameters: {total_params}")

    def _save_best_preds(self, tensors: dict, split: str = 'test') -> None:
        """Best epoch 때의 raw preds/labels를 pkl로 저장 (metric 재계산 용도)."""
        try:
            import pickle, os
            preds_dir = os.path.join(self.log_dir, 'best_preds')
            os.makedirs(preds_dir, exist_ok=True)

            save_dict = {}
            if 'preds_sites' in tensors and isinstance(tensors['preds_sites'], torch.Tensor) and tensors['preds_sites'].numel() > 0:
                import torch.nn.functional as F
                logits = tensors['preds_sites']  # (N, L-1) or (N, L-1, C)
                if logits.dim() == 3 and logits.size(-1) == 2:
                    probs = torch.softmax(logits, dim=-1)[..., 1]
                else:
                    probs = torch.sigmoid(logits.squeeze(-1))
                labels = tensors['labels_sites'][:, 1:]  # CLS 제거
                lengths = tensors.get('lengths_sites', None)
                save_dict['probs_sites']  = probs.cpu()
                save_dict['labels_sites'] = labels.cpu()
                if lengths is not None:
                    save_dict['lengths_sites'] = lengths.cpu()

            if 'preds_bind' in tensors and isinstance(tensors['preds_bind'], torch.Tensor) and tensors['preds_bind'].numel() > 0:
                save_dict['probs_bind']  = torch.sigmoid(tensors['preds_bind'].squeeze(-1)).cpu()
                save_dict['labels_bind'] = tensors['labels_bind'].cpu()

            if save_dict:
                out_path = os.path.join(preds_dir, f'{split}_preds.pkl')
                with open(out_path, 'wb') as f:
                    pickle.dump(save_dict, f)
                self.logger.info(f"Best preds saved → {out_path}")
        except Exception as e:
            self.logger.info(f"[WARN] _save_best_preds failed: {e}")

    def save_model(self, epoch: int = None, pretrain: bool = False, verbose: bool = False) -> None:
        save_model(
            model=self.model,
            dir_save=self.dir_save,
            model_name=self.model_name,
            experiment_name=self.experiment_name,
            seed=self.seed,
            rank=0,
            pretrain=pretrain,
            epoch=epoch,
            verbose=verbose
        )

    def load_model(self, epoch: int = None, pretrain: bool = False, verbose: bool = False, load_pretrain_name: str = None) -> None:
        self.model = load_model(
            model=self.model,
            dir_save=self.dir_save,
            model_name=self.model_name,
            experiment_name=self.experiment_name,
            seed=self.seed,
            device=self.device,
            pretrain=pretrain,
            load_pretrain_name=load_pretrain_name,
            epoch=epoch,
            verbose=verbose
        )

    def load_model_from_path(self, model_path: str, verbose: bool = False) -> None:
        """Load model weights from an explicit file path, skipping mismatched keys."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        model_state = self.model.state_dict()

        # Filter: only load keys that exist in both and have matching shapes
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in state_dict if k not in filtered]

        model_state.update(filtered)
        self.model.load_state_dict(model_state)

        if verbose:
            print(f"Loaded {len(filtered)}/{len(state_dict)} params from {model_path}")
            if skipped:
                print(f"  Skipped {len(skipped)} mismatched keys: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    def update_best_model(self, score: float, epoch: int, pretrain: bool = False) -> bool:
        try:
            current_best = self.best_score
            is_better = score > current_best if not pretrain else score < current_best
            if is_better:
                self.best_score = score
                self.best_epoch = epoch
                self.save_model(epoch=epoch, pretrain=pretrain, verbose=False)
                self.logger.info(f"Best model updated at epoch {epoch}: score {score:.4f}")
                self.patience = 0
                return True
            else:
                self.patience += 1
                self.logger.info(f"No improvement at epoch {epoch}. Patience: {self.patience}/{self.patience_max}. Best: {self.best_score:.4f} at epoch {self.best_epoch}")
                return False
        except Exception as e:
            self.logger.info(f"Error updating best model: {e}")
            return False

    def train(
            self, 
            epochs: int = 100, 
            earlystop: int = 20, 
            task: str = 'binding', 
            forward_mode: int = 1, 
            rc: bool = False, 
            log_name: str = 'training',
            alpha: float = 1.0,
            beta: float = 1.0,
        ) -> dict:

        self.task = task
        self.best_score = 0.0
        self.patience = 0
        self.best_epoch = 0
        self.epochs = epochs
        self.logs = {'train': {}, 'valid': {}, 'test': {}, 'extra': {}, 'final': {}}
        self.rc = rc
        self.forward_mode = forward_mode
        self.alpha = alpha
        self.beta = beta
        self.patience_max = earlystop

        # Set site class weights for unified head or sites/both task
        if self.task in ['sites', 'both'] or self.use_unified_head:
            self.site_class_weights = calculate_class_weights_from_df(
                self.train_loader.dataset.df,
                class_col_name='sites',
                length_col_name='length',
                device=self.device
            )
            self.loss_fn_sites = nn.CrossEntropyLoss(weight=self.site_class_weights, ignore_index=-100).to(device=self.device)

        self.setup_logging(log_name)
        self.logger.info(f"Starting training for model {self.model_name} on task {task} (rc: {rc}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        for epoch in range(1, epochs + 1):
            if self.verbose:
                print(f"\n=== Epoch {epoch}/{epochs} ===")
            loss_train, tensors_train, scores_train = self.step_loader(self.train_loader, epoch, is_train=True, data_type='Train')
            self.logs['train'][epoch] = {'loss': float(loss_train['loss_total']), 'scores': scores_train}
            
            with torch.no_grad():
                loss_valid, tensors_valid, scores_valid = self.step_loader(self.valid_loader, epoch, is_train=False, data_type='Valid')
                self.logs['valid'][epoch] = {'loss': float(loss_valid['loss_total']), 'scores': scores_valid}

            # Compute validation score
            score_valid = 0.0
            if self.use_unified_head:
                # Unified head: sites F1 is the main metric
                if scores_valid.get('sites') and 'f1_macro' in scores_valid['sites']:
                    score_valid = scores_valid['sites']['f1_macro']
            else:
                # Legacy approach
                if self.task in ['binding', 'both']:
                    score_valid += scores_valid['bind']['f1_macro']
                if self.task in ['sites', 'both']:
                    score_valid += scores_valid['sites']['f1_macro']
                if self.task == 'both':
                    score_valid /= 2.0

            if self.update_best_model(score_valid, epoch):
                if self.task in ['sites', 'both']:
                    self.best_threshold_site = scores_valid['sites']['threshold_used']
                self.save_model(pretrain=False, verbose=False)
                with torch.no_grad():
                    loss_test, tensors_test, scores_test = self.step_loader(self.test_loader, epoch, is_train=False, data_type='Test')
                    self.logs['test'][epoch] = {'loss': float(loss_test['loss_total']), 'scores': scores_test}
                    self._save_best_preds(tensors_test, split='test')

                    if self.extra_loader is not None:
                        loss_extra, tensors_extra, scores_extra = self.step_loader(self.extra_loader, epoch, is_train=False, data_type='Extra')
                        self.logs['extra'][epoch] = {'loss': float(loss_extra['loss_total']), 'scores': scores_extra}
            
            if self.patience >= self.patience_max:
                self.logger.info(f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}, score: {self.best_score:.4f}")
                break
            else:
                pass
        
        self.logger.info(f"Training completed. Best epoch: {self.best_epoch}, score: {self.best_score:.4f}")
        self.load_model(epoch=self.best_epoch, pretrain=False, verbose=False)
        best_model_path = os.path.join(self.dir_save, self.model_name, self.experiment_name, "epoch", str(self.best_epoch), "model.pth")
        self.logger.info(f"# Loading best model from {best_model_path}")

        with torch.no_grad():
            loss_final, tensors_final, scores_final = self.step_loader(self.test_loader, self.best_epoch, is_train=False, data_type='Final')
            self.logs['final'][self.best_epoch] = {'loss': float(loss_final['loss_total']), 'scores': scores_final}

        save_logs(logs=self.logs, log_dir=self.log_dir, log_file=log_name, verbose=self.verbose)
        return self.logs

    def evaluate(self, data_loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, Any]]:
        if self.task in ['sites', 'both'] or self.use_unified_head:
            site_class_weights = calculate_class_weights_from_df(
                data_loader.dataset.df,
                class_col_name='sites',
                length_col_name='length',
                device=self.device
            )
            self.loss_fn_sites = nn.CrossEntropyLoss(weight=site_class_weights, ignore_index=-100).to(device=self.device)

        if self.alpha is None:
            self.alpha = 0.5
        if self.beta is None:   
            self.beta = 0.5
        with torch.no_grad():
            loss_valid, tensors_valid, scores_valid = self.step_loader(data_loader, 0, is_train=False, data_type='Valid')

            # Compute validation score
            score_valid = 0.0
            if self.use_unified_head:
                if scores_valid.get('sites') and 'f1_macro' in scores_valid['sites']:
                    score_valid = scores_valid['sites']['f1_macro']
            else:
                if self.task in ['binding', 'both']:
                    score_valid += scores_valid['bind']['f1_macro']
                if self.task in ['sites', 'both']:
                    score_valid += scores_valid['sites']['f1_macro']
                if self.task == 'both':
                    score_valid /= 2.0

            if scores_valid.get('sites') and 'threshold_used' in scores_valid['sites']:
                self.best_threshold_site = scores_valid['sites']['threshold_used']

    def log_scores(self, tensor_dict):

        self.logger.info(f"-" * 80)

        s_bind = {}
        s_sites = {}

        # Compute binding scores
        compute_bind = self.task in ['bind', 'binding', 'both'] or self.use_unified_head
        if compute_bind and 'preds_bind' in tensor_dict and isinstance(tensor_dict['preds_bind'], torch.Tensor):
            if tensor_dict['preds_bind'].numel() > 0:
                s_bind = cal_score_bind(tensor_dict['preds_bind'], tensor_dict['labels_bind'])
                bind_label = "BIND (derived)" if self.use_unified_head else "BIND"
                self.logger.info(f"|    MODE   | {bind_label:^12} | ACCURACY  | {s_bind['acc']:.4f} | F1 Score | {s_bind['f1_macro']:.4f} | Positive  | {s_bind['f1_pos']:.4f} |")
                self.logger.info(f"| Precision | {s_bind['prec_macro']:.4f} | Positive | {s_bind['prec_pos']:.4f} |  Recall   | {s_bind['rec_macro']:.4f} | Positive  | {s_bind['rec_pos']:.4f} |")
                self.logger.info(f"|    MCC    | {s_bind['mcc']:.4f} | ROC  AUC | {s_bind['roc_auc']:.4f} |   AUPRC   | {s_bind['auprc']:.4f} |")

        # Compute sites scores
        compute_sites = self.task in ['sites', 'both'] or self.use_unified_head
        if compute_sites and 'preds_sites' in tensor_dict and isinstance(tensor_dict['preds_sites'], torch.Tensor):
            if tensor_dict['preds_sites'].numel() > 0:
                # labels_sites has CLS at index 0 (-100); preds_sites has CLS removed (via x[:,1:,:]).
                # Remove CLS from labels to align correctly: preds[i,j] ↔ site[j]
                s_sites = cal_score_sites(tensor_dict['preds_sites'], tensor_dict['labels_sites'][:, 1:], tensor_dict['lengths_sites'])
                self.logger.info(f"|    MODE   |  SITE  | ACCURACY  | {s_sites['acc']:.4f} | F1 Score | {s_sites['f1_macro']:.4f} | Positive  | {s_sites['f1_pos']:.4f} |")
                self.logger.info(f"| Precision | {s_sites['prec_macro']:.4f} | Positive  | {s_sites['prec_pos']:.4f} | Recall   | {s_sites['rec_macro']:.4f} | Positive  | {s_sites['rec_pos']:.4f} |")
                self.logger.info(f"|    MCC    | {s_sites['mcc']:.4f} | ROC  AUC  | {s_sites['roc_auc']:.4f} |  AUPRC   | {s_sites['auprc']:.4f} |")
                self.logger.info(f"|  Span-f1  | {s_sites['span_f1']:.4f} | Span-prec | {s_sites['span_precision']:.4f} | Span-rec | {s_sites['span_recall']:.4f} |")

        self.logger.info(f"-" * 80)

        return {'bind': s_bind, 'sites': s_sites}

    def step_loader(self, data_loader, epoch, is_train=True, data_type='Train'):
        time_start = time.time()
        self.model.train() if is_train else self.model.eval()
        loss_dict = {'loss_total': 0.0, 'loss_bind': 0.0, 'loss_sites': 0.0}
        tensor_dict = {'labels_bind': [], 'labels_sites': [], 'preds_bind': [], 'preds_sites': [], 'lengths': [], 
                       'lengths_sites': []}
        
        scores = {'bind': {}, 'sites': {}}
        for i, data in enumerate(data_loader):
            loss = torch.tensor(0.0, device=self.device, requires_grad=False)
            lengths = data['length'].to(self.device)
            if self.task in ['binding', 'both']:
                label_bind = data['binding'].to(self.device)
            if self.task in ['sites', 'both']:
                label_sites = data['sites'].to(self.device)
            if is_train:
                self.optimizer.zero_grad()

            target, target_mask = self.forward_target(data)
            emb, mask = self.forward(data)
            if self.model.is_cross_attention:
                emb, attn_map = self.forward_cross_attention(emb, target, target_mask)
            else:
                attn_map = None

            # Unified site-first approach: sites prediction is main, binding is derived
            if self.use_unified_head:
                bind_mask = data['binding'].squeeze(-1).bool().to(self.device)

                if bind_mask.any():
                    emb_sites = emb[bind_mask]
                    mask_sites = mask[bind_mask]
                    target_filtered = target[bind_mask]
                    label_sites_filtered = label_sites[bind_mask]
                    lengths_filtered = lengths[bind_mask]

                    # Forward through unified head - returns dict
                    unified_output = self.forward_task(emb_sites, target_filtered, mask=mask_sites)
                    pred_sites_filtered = unified_output['sites_logits']  # [B, L, 1]
                    pred_bind_from_sites = unified_output['binding_logits']  # [B, 1]

                    # Site loss (main task)
                    loss_sites = batched_focal_loss(
                        pred_sites_filtered.squeeze(-1),  # [B, L]
                        label_sites_filtered[:, 1:],
                        lengths_filtered,
                        class_weights=self.site_class_weights,
                        gamma=2.0,
                        ignore_index=-100,
                        sample_ratio=0.1
                    )
                    loss += loss_sites

                    # Derived binding prediction (always set for unified head)
                    pred_bind = pred_bind_from_sites  # [B_filtered, 1]

                    # Binding loss (derived, optional for 'both' task)
                    if self.task == 'both':
                        label_bind_filtered = label_bind[bind_mask]
                        loss_bind = cal_loss_bind(self.loss_fn, pred_bind, label_bind_filtered)
                        loss += self.alpha * loss_bind
                    else:
                        loss_bind = torch.tensor(0.0).to(self.device)
                else:
                    loss_sites = torch.tensor(0.0).to(self.device)
                    loss_bind = torch.tensor(0.0).to(self.device)
                    pred_sites_filtered = torch.tensor([]).to(self.device)
                    pred_bind = torch.tensor([]).to(self.device)
                    label_sites_filtered = torch.tensor([]).to(self.device)
                    lengths_filtered = torch.tensor([]).to(self.device)

            else:
                # Legacy approach: separate binding and sites heads
                if self.task in ['binding', 'both']:
                    pred_bind = self.forward_task(emb, target, task='binding')
                    loss_bind = cal_loss_bind(self.loss_fn, pred_bind, label_bind)
                    loss += self.alpha * loss_bind

                if self.task in ['sites', 'both']:
                    bind_mask = data['binding'].squeeze(-1).bool().to(self.device)

                    if bind_mask.any():
                        emb_sites = emb[bind_mask] if bind_mask.any() else emb
                        mask = mask[bind_mask] if bind_mask.any() else mask
                        target = target[bind_mask] if bind_mask.any() else target
                        target_mask = target_mask[bind_mask] if bind_mask.any() else target_mask
                        label_sites_filtered = label_sites[bind_mask]
                        lengths_filtered = lengths[bind_mask]

                        pred_sites_filtered = self.forward_task(emb_sites, target, task='sites')

                        loss_sites = batched_focal_loss(
                            pred_sites_filtered,
                            label_sites_filtered[:, 1:],
                            lengths_filtered,
                            class_weights=self.site_class_weights,
                            gamma=2.0,
                            ignore_index=-100,
                            sample_ratio=0.1
                        )
                    else:
                        loss_sites = torch.tensor(0.0).to(self.device)
                        pred_sites_filtered = torch.tensor([]).to(self.device)
                        label_sites_filtered = torch.tensor([]).to(self.device)
                        lengths_filtered = torch.tensor([]).to(self.device)

                    loss += self.beta * loss_sites

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            loss_dict['loss_total'] += loss.item()

            # Collect predictions based on approach
            if self.use_unified_head:
                # Unified head: binding is derived from sites
                # Always collect sites predictions
                loss_dict['loss_sites'] += loss_sites.item()
                if pred_sites_filtered.numel() > 0:
                    tensor_dict['preds_sites'].append(pred_sites_filtered.detach().cpu())
                    tensor_dict['labels_sites'].append(label_sites_filtered.detach().cpu())
                    tensor_dict['lengths_sites'].append(lengths_filtered.detach().cpu())

                # Always collect derived binding predictions (for reporting)
                loss_dict['loss_bind'] += loss_bind.item()
                if pred_bind.numel() > 0:
                    tensor_dict['preds_bind'].append(pred_bind.detach().cpu())
                    # Labels for binding: all 1s since we only process binding=1 samples
                    bind_labels = torch.ones(pred_bind.size(0), dtype=torch.long)
                    tensor_dict['labels_bind'].append(bind_labels.cpu())
            else:
                # Legacy separate heads
                if self.task in ['binding', 'both']:
                    loss_dict['loss_bind'] += loss_bind.item()
                    tensor_dict['preds_bind'].append(pred_bind.detach().cpu())
                    tensor_dict['labels_bind'].append(label_bind.detach().cpu())

                if self.task in ['sites', 'both']:
                    loss_dict['loss_sites'] += loss_sites.item()
                    tensor_dict['preds_sites'].append(pred_sites_filtered.detach().cpu())
                    tensor_dict['labels_sites'].append(label_sites_filtered.detach().cpu())
                    tensor_dict['lengths_sites'].append(lengths_filtered.detach().cpu())

            tensor_dict['lengths'].append(lengths.detach().cpu())

            if self.verbose:
                print(f"\r [Batch {i+1}/{len(data_loader)}] Loss: {loss.item():.4f}", end='')

        if self.verbose:
            print(f"\r                                                  ", end='')
            print(f'\r', end='')
        time_end = time.time()
        loss_dict['loss_total'] /= len(data_loader)

        # Concatenate collected tensors
        if self.use_unified_head:
            # Unified head always has both sites and derived binding
            if tensor_dict['labels_sites']:
                tensor_dict['labels_sites'] = torch.cat(tensor_dict['labels_sites']).cpu()
                tensor_dict['preds_sites'] = torch.cat(tensor_dict['preds_sites']).cpu()
                tensor_dict['lengths_sites'] = torch.cat(tensor_dict['lengths_sites']).cpu()
            if tensor_dict['labels_bind']:
                tensor_dict['labels_bind'] = torch.cat(tensor_dict['labels_bind']).cpu()
                tensor_dict['preds_bind'] = torch.cat(tensor_dict['preds_bind']).cpu()
            loss_dict['loss_sites'] /= len(data_loader)
            loss_dict['loss_bind'] /= len(data_loader)
        else:
            # Legacy approach
            if self.task in ['binding', 'both']:
                tensor_dict['labels_bind'] = torch.cat(tensor_dict['labels_bind']).cpu()
                tensor_dict['preds_bind'] = torch.cat(tensor_dict['preds_bind']).cpu()
                loss_dict['loss_bind'] /= len(data_loader)

            if self.task in ['sites', 'both']:
                tensor_dict['labels_sites'] = torch.cat(tensor_dict['labels_sites']).cpu()
                tensor_dict['preds_sites'] = torch.cat(tensor_dict['preds_sites']).cpu()
                tensor_dict['lengths_sites'] = torch.cat(tensor_dict['lengths_sites']).cpu()
                loss_dict['loss_sites'] /= len(data_loader)

        tensor_dict['lengths'] = torch.cat(tensor_dict['lengths']).cpu()

        if self.task in ['both']:
            self.logger.info(f"[ {data_type} ] Epoch {epoch} | Loss: {loss_dict['loss_total']:.4f} | Loss Bind: {loss_dict['loss_bind']:.4f} | Loss Sites: {loss_dict['loss_sites']:.4f} | Time: {time_end - time_start:.2f}s |")
        else:
            self.logger.info(f"[ {data_type} ] Epoch {epoch} | Loss: {loss_dict['loss_total']:.4f} | Time: {time_end - time_start:.2f}s")

        scores = self.log_scores(tensor_dict=tensor_dict)

        return loss_dict, tensor_dict, scores

    def forward(self, data):
        x, x_rc, x_mask, x_rc_mask = self.get_data(data)

        if self.model_name.lower() in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm'] and hasattr(self, 'model_pt'):
            with torch.no_grad():
                x = self.model_pt(x, x_mask)
                x = x['last_hidden_state']
                if self.rc:
                    x_rc = self.model_pt(x_rc, x_rc_mask)
                    x_rc = x_rc['last_hidden_state']

        emb, emb_rc = self.model.forward(x, x_mask, x_rc, x_rc_mask)

        return emb, x_mask

    def _apply_interaction(self, emb, target):
        """Apply interaction mechanism between circRNA embedding and miRNA target."""
        if self.interaction == 'cross_attention':
            return emb  # Already fused via forward_cross_attention

        target_proj = self.model.get_target_projected(target, mode='cls')  # [B, D]
        seq_len = emb.size(1)
        target_proj = target_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]

        if self.interaction == 'elementwise':
            return emb * target_proj  # [B, L, D]
        else:  # 'concat'
            return torch.cat((emb, target_proj), dim=-1)  # [B, L, 2D]

    def forward_task(self, emb, target, task: str = 'sites', mask: torch.Tensor = None):
        # Unified site-first approach
        if self.use_unified_head:
            x = self._apply_interaction(emb, target)
            x = x[:, 1:, :]  # Remove CLS token position

            # Create mask for valid positions (excluding padding)
            if mask is not None:
                site_mask = mask[:, 1:]  # Remove CLS position from mask
            else:
                site_mask = None

            output = self.model.unified_site_head(x, mask=site_mask)
            return output  # Returns dict with 'sites_logits', 'binding_logits', 'sites_probs'

        # Legacy separate head approach
        if task == 'binding':
            if self.model_name.lower() in ['transformer', 'hymba']:
                return self.model.binding_head(emb[:, 0, :])
            else:
                return self.model.binding_head(emb[:, 1:, :].mean(dim=1))
        elif task == 'sites':
            x = self._apply_interaction(emb, target)
            return self.model.binding_site_head(x[:, 1:, :])
        else:
            raise ValueError(f"Task '{task}' not recognized.")

    def get_data(self, data):
        x = data['circRNA'].to(self.device)
        x_rc = data['circRNA_rc'].to(self.device)
        x_mask = data['circRNA_mask'].to(self.device)
        x_rc_mask = data['circRNA_rc_mask'].to(self.device)
        return x, x_rc, x_mask, x_rc_mask

    def forward_target(self, data):
        x_target = data['target'].to(self.device)
        target_mask = data['target_mask'].to(self.device)
        with torch.no_grad():
            target = self.model_target(x_target, target_mask)
        return target['last_hidden_state'], target_mask

    def forward_cross_attention(self, emb, target, target_mask=None):
        target_proj = self.model.get_target_projected(target, mode='None')
        emb_out, attn_maps = self.model.cross_attention(emb, target_proj, target_proj, target_mask)
        return emb_out, attn_maps

    def inference(self, data_loader):
        """
        Run inference on the entire dataset and return predictions and targets.
        """
        self.model.eval()
        all_preds = {'binding': [], 'sites': [], 'lengths': [], 'lengths_sites': [], 'attn_maps': []}
        all_targets = {'binding': [], 'sites': []}
        all_lengths = []
        with torch.no_grad():
        
            time_start = time.time()
            self.model.eval()
            loss_dict = {'loss_total': 0.0, 'loss_bind': 0.0, 'loss_sites': 0.0}
            tensor_dict = {'labels_bind': [], 'labels_sites': [], 'preds_bind': [], 'preds_sites': [], 'lengths': [], 
                           'lengths_sites': [], 'attn_maps': []}
            scores = {'bind': {}, 'sites': {}}


            for i, data in enumerate(data_loader):
                loss = torch.tensor(0.0, device=self.device, requires_grad=False)

                lengths = data['length'].to(self.device)
                tensor_dict['lengths'].append(lengths.detach().cpu())

                target, target_mask = self.forward_target(data)
                emb, mask = self.forward(data)
                if self.model.is_cross_attention:
                    emb, attn_map = self.forward_cross_attention(emb, target, target_mask)
                else:
                    attn_map = None
                tensor_dict['attn_maps'].append(attn_map.detach().cpu() if attn_map is not None else None)

                if self.task in ['binding', 'both']:
                    pred_bind = self.forward_task(emb, target, task='binding')
                    tensor_dict['preds_bind'].append(pred_bind.detach().cpu())

                if self.task in ['sites', 'both']:    
                    pred_sites = self.forward_task(emb, target, task='sites')
                    tensor_dict['preds_sites'].append(pred_sites.detach().cpu())
            time_end = time.time()

            tensor_dict['all_lengths'] = torch.cat(tensor_dict['lengths']).cpu()
            tensor_dict['all_attn_maps'] = torch.cat(tensor_dict['attn_maps']).cpu()
            if self.task in ['binding', 'both']:
                tensor_dict['preds_bind'] = torch.cat(tensor_dict['preds_bind']).cpu()
                # tensor_dict['labels_bind'] = torch.cat(tensor_dict['labels_bind']).cpu()
                all_preds['binding'].append(tensor_dict['preds_bind'])
                # all_targets['binding'].append(tensor_dict['labels_bind'])
            
            if self.task in ['sites', 'both']:
                tensor_dict['preds_sites'] = torch.cat(tensor_dict['preds_sites']).cpu()
                all_preds['sites'].append(tensor_dict['preds_sites'])
            all_preds['lengths'].append(tensor_dict['all_lengths'])
            all_preds['attn_maps'].append(tensor_dict['all_attn_maps'])

        return all_preds
            
    
    def _set_loss_fn(self):
        self.loss_fn_mlm = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_ntp = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_ssp = nn.CrossEntropyLoss(ignore_index=-100)
        self.ss_labels_weights = calculate_class_weights_from_df(
           self.train_self.dataset.df,
           class_col_name='ss_labels',
           length_col_name='length', 
           device=self.device
        )
        self.loss_fn_ss_labels = nn.CrossEntropyLoss(weight=self.ss_labels_weights, ignore_index=-100)
        self.loss_fn_ss_labels_multi = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_pairing = nn.BCEWithLogitsLoss()
        task_pt = [name for name, value in self.info_pt['tasks'].items() if value]
        self.loss_uncertainty = UncertaintyWeightingLoss(task_names=task_pt)
    
    def pretrain(
            self,
            epochs=100,
            earlystop=20,
            mask_ratio=0.15,
            mlm=False,
            ntp=False,
            ssp=False,
            ss_labels=False,
            ss_labels_multi=False,
            pairing=False,
            cpcl=False,
            bsj_mlm=False,
            ss_cl=False,
            icl=False,
            icl_mlm=False,
            mcl=False,
            rc=False,
            log_name='pretrain',
            verbose=True,
            ssp_vocab_size: Optional[int] = None
        ):
        self.patience = 0
        self.patience_max = earlystop
        self.best_epoch = 0
        self.best_score = float('inf')  # Lower loss is better in pretraining.
        self.rc = rc

        if ssp:
            if ssp_vocab_size is None:
                ssp_vocab_size = 4

        if ss_cl:
            # SS-pair CL requires SS embedding and contrastive projection head
            ss_vocab_size = 4  # PAD, (, ), .
            self.model._set_ss_embedding(ss_vocab_size)
            self.model._set_proj_contrastive()

        self.n_task = sum([mlm, ntp, ssp, icl, icl_mlm, mcl, pairing, ss_labels, ss_labels_multi, cpcl, bsj_mlm, ss_cl])

        self.task_masking_config = {
            'ss_labels': {'use_masking': False, 'mask_ratio': 0.0},
            'ss_labels_multi': {'use_masking': True, 'mask_ratio': 0.15},
            'ssp': {'use_masking': True, 'mask_ratio': 0.15},
            'pairing': {'use_masking': False, 'mask_ratio': 0.0},
        }

        if self.n_task == 0:
            raise ValueError("At least one pretraining task must be enabled (mlm, ntp, ssp, pairing, ss_label).")
        self.mask_ratio = mask_ratio
        self.verbose = verbose

        self.logs_pretrain = {'train': {}, 'valid': {}}
        self.log_dir = os.path.join(self.dir_log, self.model_name, f'{self.experiment_name}', str(self.seed))
        self.log_file = os.path.join(self.log_dir, f'{log_name}.log')

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger()

        self.info_pt = {
            'tasks': {
                'mlm': mlm,
                'ntp': ntp,
                'ssp': ssp,
                'ss_labels': ss_labels,
                'ss_labels_multi': ss_labels_multi,
                'pairing': pairing,
                'cpcl': cpcl,
                'bsj_mlm': bsj_mlm,
                'ss_cl': ss_cl,
            },
            'type': {
                'self': mlm or ntp or mcl or ssp or ss_labels or ss_labels_multi or pairing or cpcl or bsj_mlm or ss_cl,
            },
            'mask_ratio': mask_ratio,
            'ssp_vocab_size': ssp_vocab_size,
        }

        self.logger.info('-' * 50)
        self.logger.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Starting pretraining for {epochs} epochs...")
        self.logger.info(f"Number of tasks: {self.n_task}")
        self.logger.info(str(self.info_pt['tasks']))
        self._set_loss_fn()

        # Add uncertainty weighting parameters to optimizer if multi-task
        if self.n_task > 1:
            self.loss_uncertainty = self.loss_uncertainty.to(self.device)
            self.optimizer.add_param_group({
                'params': self.loss_uncertainty.parameters()
            })

        # Cosine LR scheduler: decays lr to 0 over training → forces convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        for epoch in range(1, epochs + 1):
            self.logger.info('-' * 50)
            self.logger.info(f"[Epoch {epoch}/{epochs}]")
            self.logs_pretrain['train'][epoch] = {}

            if self.info_pt['type']['self']:
                losses_self, times_self = self.epoch_self_pretrain(is_train=True)
                self.log_epoch_results(epoch, losses_self, times_self, 'train')

            scheduler.step()
            self.logger.info(f"LR: {scheduler.get_last_lr()[0]:.2e}")

            with torch.no_grad():
                loss_valid = 0.0
                self.logs_pretrain['valid'][epoch] = {}
                if self.info_pt['type']['self']:
                    losses_self_valid, times_self_valid = self.epoch_self_pretrain(is_train=False)
                    self.log_epoch_results(epoch, losses_self_valid, times_self_valid, 'valid')

                    loss_valid += losses_self_valid['total']
                    del losses_self_valid, times_self_valid
                    torch.cuda.empty_cache()
                    import gc; gc.collect()

            if self.update_best_model(loss_valid, epoch, pretrain=True):
                self.best_epoch = epoch
                self.save_model(pretrain=True, verbose=False)

            if self.patience >= self.patience_max:
                message = (f"Early stopping at epoch {epoch}. Best validation loss: {self.best_score:.4f} "
                           f"at epoch {self.best_epoch}")
                self.logger.info(message)
                break

        self.load_model(pretrain=True)
        self.logger.info(f"Loaded best pre-trained model from epoch {self.best_epoch} with validation loss: {self.best_score:.4f}")
        self.logger.info("Pretraining completed!\n")

        save_logs(logs=self.logs_pretrain, log_dir=self.log_dir, log_file=log_name, verbose=self.verbose)

    def epoch_self_pretrain(self, is_train=True, phase=None):
        losses = {'mlm': 0.0, 'ntp': 0.0, 'ssp': 0.0, 'ss_labels': 0.0, 'ss_labels_multi': 0.0, 'pairing': 0.0, 'cpcl': 0.0, 'bsj_mlm': 0.0, 'ss_cl': 0.0, 'total': 0.0}
        losses_batch = {}
        times = {'mlm': 0.0, 'ntp': 0.0, 'ssp': 0.0, 'ss_labels': 0.0, 'ss_labels_multi': 0.0, 'pairing': 0.0, 'cpcl': 0.0, 'bsj_mlm': 0.0, 'ss_cl': 0.0, 'total': 0.0}
        dataloader = self.train_self if is_train else self.valid_self

        phase = phase if phase is not None else 'TRAIN' if is_train else 'VALID'
        times_start = time.time()
        
        for i, data in enumerate(dataloader):
            if is_train:
                self.optimizer.zero_grad()

            if self.info_pt['tasks']['mlm']:
                t_mlm = time.time()
                loss_mlm = self.forward_mlm(data, mask_ratio=self.info_pt['mask_ratio'])
                losses_batch['mlm'] = loss_mlm
                losses['mlm'] += loss_mlm.item()
                times['mlm'] += time.time() - t_mlm
            else:
                loss_mlm = 0.0

            if self.info_pt['tasks']['ntp']:
                t_ntp = time.time()
                loss_ntp = self.forward_ntp(data)
                losses_batch['ntp'] = loss_ntp
                losses['ntp'] += loss_ntp.item()
                times['ntp'] += time.time() - t_ntp
            else:
                loss_ntp = 0.0

            if self.info_pt['tasks']['ssp']:
                t_ssp = time.time()
                loss_ssp = self.forward_ssp(data)
                losses_batch['ssp'] = loss_ssp
                losses['ssp'] += loss_ssp.item()
                times['ssp'] += time.time() - t_ssp
            else:
                loss_ssp = 0.0

            if self.info_pt['tasks']['ss_labels']:
                t_ss_label = time.time()
                loss_ss_labels = self.forward_ss_labels(data)
                losses_batch['ss_labels'] = loss_ss_labels
                losses['ss_labels'] += loss_ss_labels.item()
            else:
                loss_ss_labels = 0.0

            if self.info_pt['tasks']['ss_labels_multi']:
                t_ss_label_multi = time.time()
                loss_ss_labels_multi = self.forward_ss_labels_multi(data)
                losses_batch['ss_labels_multi'] = loss_ss_labels_multi
                losses['ss_labels_multi'] += loss_ss_labels_multi.item()
                times['ss_labels_multi'] += time.time() - t_ss_label_multi
            else:
                loss_ss_labels_multi = 0.0

            if self.info_pt['tasks']['pairing']:
                t_pairing = time.time()
                loss_pairing = self.forward_pairing(data)
                losses_batch['pairing'] = loss_pairing
                losses['pairing'] += loss_pairing.item()
                times['pairing'] += time.time() - t_pairing
            else:
                loss_pairing = 0.0

            if self.info_pt['tasks']['cpcl']:
                t_cpcl = time.time()
                loss_cpcl = self.forward_cpcl(data)
                losses_batch['cpcl'] = loss_cpcl
                losses['cpcl'] += loss_cpcl.item()
                times['cpcl'] += time.time() - t_cpcl
            else:
                loss_cpcl = 0.0

            if self.info_pt['tasks']['bsj_mlm']:
                t_bsj_mlm = time.time()
                loss_bsj_mlm = self.forward_bsj_mlm(data, mask_ratio=self.info_pt['mask_ratio'])
                losses_batch['bsj_mlm'] = loss_bsj_mlm
                losses['bsj_mlm'] += loss_bsj_mlm.item()
                times['bsj_mlm'] += time.time() - t_bsj_mlm
            else:
                loss_bsj_mlm = 0.0

            if self.info_pt['tasks']['ss_cl']:
                t_ss_cl = time.time()
                loss_ss_cl = self.forward_ss_cl(data)
                losses_batch['ss_cl'] = loss_ss_cl
                losses['ss_cl'] += loss_ss_cl.item()
                times['ss_cl'] += time.time() - t_ss_cl
            else:
                loss_ss_cl = 0.0

            # total_loss = loss_mlm + loss_ntp + loss_ssp + loss_ss_labels + loss_ss_labels_multi + loss_pairing
            # total_loss = total_loss / self.n_task if self.n_task > 0 else total_loss
            total_loss, individual_losses = self.loss_uncertainty(losses_batch)

            total_time = time.time() - times_start
            if is_train:
                total_loss.backward()
                self.optimizer.step()
            losses['total'] = total_loss.item() if i == 0 else losses.get('total', 0) + total_loss.item()
            times['total'] += total_time

            if self.verbose:
                print(f"\r[Pretraining - Batch {i+1}/{len(dataloader)}] Loss: {total_loss.item():.4f}", end='')

        if self.verbose:
            print(f"\r                                                  ", end='')
            print(f'\r', end='')

        for key in losses:
            losses[key] /= len(dataloader)
            times[key] /= len(dataloader)

        return losses, times

    def apply_masking_for_task(self, preds, targets, task_name: str):
        config = self.task_masking_config.get(task_name, {})
        use_masking = config.get('use_masking', False)
        mask_ratio = config.get('mask_ratio', 0.0)

        if preds.dim() == 3:  # [B, L, C]
            preds = preds.view(-1, preds.size(-1))  # [B*L, C]
        else:
            preds = preds.view(-1)

        targets = targets.view(-1)

        if not use_masking or mask_ratio <= 0.0:
            return preds, targets

        mask = torch.rand_like(targets.float()) < mask_ratio
        if mask.sum() == 0:
            return torch.tensor([]).to(preds.device), torch.tensor([]).to(targets.device)

        return preds[mask], targets[mask]


    def forward_mlm(self, data, mask_ratio=0.15):
        x = data['circRNA'].to(self.device)
        mask = data['circRNA_mask'].to(self.device)
        x_rc = data['circRNA_rc'].to(self.device)
        mask_rc = data['circRNA_rc_mask'].to(self.device)
        x_mlm, y_mlm = create_mlm_inputs_and_labels(x, mask_ratio, device=self.device, attention_mask=mask)

        if self.rc:
            x_mlm_rc, y_mlm_rc = create_mlm_inputs_and_labels(x_rc, mask_ratio, device=self.device, attention_mask=mask_rc)
        else:
            x_mlm_rc, y_mlm_rc = None, None
        emb = self.model.embedding(x_mlm)
        emb = self.model.token_dropout(emb, x_mlm)

        if self.rc:
            emb_rc = self.model.embedding(x_mlm_rc)
            emb_rc = self.model.token_dropout(emb_rc, x_mlm_rc)
        else:
            emb_rc = None

        out, out_rc = self.model.backbone(emb, mask, emb_rc, mask_rc)
        y_hat = self.model.mlm_head(out)

        y_hat = y_hat.contiguous().view(-1, y_hat.size(-1))
        y_mlm = y_mlm.contiguous().view(-1)

        loss = self.loss_fn_mlm(y_hat, y_mlm)
        if self.rc:
            y_hat_rc = self.model.mlm_head(out_rc)
            y_hat_rc = y_hat_rc.contiguous().view(-1, y_hat_rc.size(-1))
            y_mlm_rc = y_mlm_rc.contiguous().view(-1)
            loss += self.loss_fn_mlm(y_hat_rc, y_mlm_rc)
        return loss

    def forward_ntp(self, data):
        x = data['circRNA'].to(self.device)
        mask = data['circRNA_mask'].to(self.device)
        x_rc = data['circRNA_rc'].to(self.device)
        mask_rc = data['circRNA_rc_mask'].to(self.device)
        x_ntp, y_ntp = create_ntp_inputs_and_labels_with_mask(x, mask, device=self.device)

        if self.rc:
            x_ntp_rc, y_ntp_rc = create_ntp_inputs_and_labels_with_mask(x_rc, mask_rc, device=self.device)
        else:
            x_ntp_rc, y_ntp_rc = None, None
        emb = self.model.embedding(x_ntp)
        emb = self.model.token_dropout(emb, x_ntp)

        if self.rc:
            emb_rc = self.model.embedding(x_ntp_rc)
            emb_rc = self.model.token_dropout(emb_rc, x_ntp_rc)
        else:
            emb_rc = None

        out, out_rc = self.model.backbone(emb, mask, emb_rc, mask_rc)
        y_hat = self.model.ntp_head(out)
        y_hat = y_hat.contiguous().view(-1, y_hat.size(-1))
        y_ntp = y_ntp.contiguous().view(-1)
        loss = self.loss_fn_ntp(y_hat, y_ntp)

        if self.rc:
            y_hat_rc = self.model.ntp_head(out_rc)
            y_hat_rc = y_hat_rc.contiguous().view(-1, y_hat_rc.size(-1))
            y_ntp_rc = y_ntp_rc.contiguous().view(-1)
            loss += self.loss_fn_ntp(y_hat_rc, y_ntp_rc)

        return loss

    def forward_ssp(self, data):

        x = data['circRNA'].to(self.device)
        mask = data['circRNA_mask'].to(self.device)
        target = data['structure'].to(self.device)

        emb = self.model.embedding(x)
        emb = self.model.token_dropout(emb, x)

        out, _ = self.model.backbone(emb, mask, None, None)
        
        pred = self.model.ssp_head(out)  
        pred, target = self.apply_masking_for_task(pred, target, 'ssp')

        loss = self.loss_fn_ssp(pred.contiguous().view(-1, pred.size(-1)), target.contiguous().view(-1).long())

        return loss
    
    def forward_ss_labels(self, data):
        x = data['circRNA'].to(self.device)
        mask = data['circRNA_mask'].to(self.device)
        labels = data['ss_labels'].to(self.device).float()  # ensure float

        emb = self.model.embedding(x)
        emb = self.model.token_dropout(emb, x)
        out, _ = self.model.backbone(emb, mask, None, None)

        pred_ss = self.model.ss_labels_head(out).squeeze(-1)  # [B, L]

        pred_ss, labels = self.apply_masking_for_task(pred_ss, labels, 'ss_labels')

        valid_mask = (labels == 0) | (labels == 1)
        pred_ss = pred_ss[valid_mask]
        labels = labels[valid_mask]

        if pred_ss.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        loss = self.loss_fn_ss_labels(pred_ss, labels.long())
        return loss

    
    def forward_ss_labels_multi(self, data):
        x = data['circRNA'].to(self.device)
        mask = data['circRNA_mask'].to(self.device)
        labels = data['ss_labels_multi'].to(self.device)

        emb = self.model.embedding(x)
        emb = self.model.token_dropout(emb, x)

        out, _ = self.model.backbone(emb, mask, None, None)
        pred = self.model.ss_labels_multi_head(out)

        pred, labels = self.apply_masking_for_task(pred, labels, 'ss_labels_multi')

        if pred.numel() == 0 or labels.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        labels = labels.long()
        loss = self.loss_fn_ss_labels_multi(pred.view(-1, pred.size(-1)), labels.view(-1))
        return loss

    def forward_pairing(self, data):
        x = data['circRNA'].to(self.device)
        mask = data['circRNA_mask'].to(self.device).float()   # [B, L]
        pairing = data['pairing_masked'].to(self.device)      # [B, L, L], PAD zeroed

        emb = self.model.embedding(x)
        emb = self.model.token_dropout(emb, x)

        out, _ = self.model.backbone(emb, mask, None, None)

        pairing_out = self.model.pairing_head(out)            # [B, L, L] raw logits

        # Mask out PAD positions — only compute loss over real sequence positions
        mask_2d = mask.unsqueeze(2) * mask.unsqueeze(1)       # [B, L, L]
        loss_raw = F.binary_cross_entropy_with_logits(pairing_out, pairing, reduction='none')
        loss = (loss_raw * mask_2d).sum() / mask_2d.sum().clamp(min=1)
        return loss

    def forward_cpcl(self, data, temperature=0.1, n_sample=64):
        """
        Circular Permutation Contrastive Learning (CPCL) — token-level version.

        Previous bug: mean pooling is permutation-invariant by definition,
        so the model could trivially solve the task without learning anything.

        Fix: position-aligned token-level contrastive learning.
        - Encode original x and shifted x_perm separately.
        - Align x_perm representations back to original positions:
            x_perm[j] = x[(j + shift) % L]  →  out_perm_aligned[i] = out_perm[(i - shift) % L]
        - Now out[i] and out_perm_aligned[i] are representations of the same nucleotide
          from two different linearization starting points.
        - InfoNCE on sampled positions: positive = (out[b,i], out_perm_aligned[b,i]),
          negatives = all other (b', j) pairs in the batch.
        """
        x = data['circRNA'].to(self.device)
        mask = data['circRNA_mask'].to(self.device)
        B, L = x.shape

        # Create circular permutation (shift by random amount per sample)
        shift = torch.randint(1, L, (B,), device=self.device)
        x_perm = torch.stack([
            torch.cat([x[i, s:], x[i, :s]]) for i, s in enumerate(shift)
        ])

        # Encode both sequences
        emb = self.model.embedding(x)
        emb_perm = self.model.embedding(x_perm)
        out, _      = self.model.backbone(emb,      mask, None, None)  # [B, L, D]
        out_perm, _ = self.model.backbone(emb_perm, mask, None, None)  # [B, L, D]

        # Align perm back to original positions:
        # x_perm[j] = x[(j+shift)%L]  →  representation of x[i] in perm = out_perm[(i-shift)%L]
        idx = torch.arange(L, device=self.device).unsqueeze(0)          # [1, L]
        aligned_idx = (idx - shift.unsqueeze(1)) % L                    # [B, L]
        out_perm_aligned = out_perm.gather(
            1, aligned_idx.unsqueeze(-1).expand(-1, -1, out_perm.size(-1))
        )  # [B, L, D]

        # Sample M random positions for memory efficiency
        M = min(n_sample, L)
        pos = torch.randperm(L, device=self.device)[:M]
        z1 = out[:, pos]               # [B, M, D]
        z2 = out_perm_aligned[:, pos]  # [B, M, D]

        # Project (2-layer MLP) and normalize
        z1 = F.normalize(self.model.proj_contrastive(z1).view(B * M, -1), dim=-1)
        z2 = F.normalize(self.model.proj_contrastive(z2).view(B * M, -1), dim=-1)

        # InfoNCE: (z1[k], z2[k]) is positive pair for k = b*M + m
        logits = torch.matmul(z1, z2.T) / temperature  # [B*M, B*M]
        labels = torch.arange(B * M, device=self.device)
        return F.cross_entropy(logits, labels)

    def forward_ss_cl(self, data, temperature=0.07):
        """
        SS-pair Contrastive Learning (SS-CL).

        Positive pair: same circRNA sequence with two different secondary structure predictions.
        Negative pair: different circRNA sequences.

        Requires pair_mode=True in CircRNASelfDataset (df_circ_ss_5 with ~5 SS per sequence).
        The model encodes sequence + SS via: emb = seq_embedding(x) + ss_embedding(ss_tokens).
        """
        x    = data['circRNA'].to(self.device)       # [B, L]
        mask = data['circRNA_mask'].to(self.device)   # [B, L]
        ss1  = data['structure_1'].to(self.device)    # [B, L] — SS view 1
        ss2  = data['structure_2'].to(self.device)    # [B, L] — SS view 2

        # Encode: sequence embedding + SS embedding (additive conditioning)
        emb1 = self.model.embedding(x) + self.model.ss_embedding(ss1)  # [B, L, D]
        emb2 = self.model.embedding(x) + self.model.ss_embedding(ss2)  # [B, L, D]

        out1, _ = self.model.backbone(emb1, mask, None, None)  # [B, L, D]
        out2, _ = self.model.backbone(emb2, mask, None, None)  # [B, L, D]

        # Sequence-level pooling: mean over non-padded positions
        mask_f = mask.float().unsqueeze(-1)            # [B, L, 1]
        z1 = (out1 * mask_f).sum(1) / mask_f.sum(1)   # [B, D]
        z2 = (out2 * mask_f).sum(1) / mask_f.sum(1)   # [B, D]

        # Project (shared 2-layer MLP) and normalize
        z1 = F.normalize(self.model.proj_contrastive(z1), dim=-1)  # [B, D]
        z2 = F.normalize(self.model.proj_contrastive(z2), dim=-1)  # [B, D]

        # Symmetric InfoNCE: (z1[b], z2[b]) are positive pairs
        B = z1.size(0)
        logits = torch.matmul(z1, z2.T) / temperature  # [B, B]
        labels = torch.arange(B, device=self.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss

    def forward_bsj_mlm(self, data, mask_ratio=0.15, bsj_focus_ratio=0.5):
        """
        BSJ-focused MLM: Masked Language Modeling with focus on Back-Splice Junction region.
        The BSJ in circRNA is at the junction of start and end of the linearized sequence.
        """
        x = data['circRNA'].to(self.device)
        mask = data['circRNA_mask'].to(self.device)
        B, L = x.shape

        # Create BSJ-focused mask: higher probability near the ends (BSJ region)
        n_mask = int(L * mask_ratio)
        n_bsj_mask = int(n_mask * bsj_focus_ratio)
        n_random_mask = n_mask - n_bsj_mask

        # BSJ region: first and last 10% of sequence
        bsj_region_size = max(int(L * 0.1), 5)

        x_mlm = x.clone()
        y_mlm = torch.full_like(x, -100)  # -100 is ignore index

        for i in range(B):
            seq_len = mask[i].sum().item()
            if seq_len < 10:
                continue

            # BSJ region indices (start and end)
            bsj_start_indices = list(range(1, min(bsj_region_size + 1, seq_len)))
            bsj_end_indices = list(range(max(1, seq_len - bsj_region_size), seq_len))
            bsj_indices = list(set(bsj_start_indices + bsj_end_indices))

            # Sample BSJ mask positions
            n_bsj = min(n_bsj_mask, len(bsj_indices))
            if n_bsj > 0:
                bsj_mask_pos = torch.tensor(bsj_indices)[torch.randperm(len(bsj_indices))[:n_bsj]]
            else:
                bsj_mask_pos = torch.tensor([], dtype=torch.long)

            # Sample random mask positions (excluding BSJ already masked)
            non_bsj_indices = [j for j in range(1, seq_len) if j not in bsj_mask_pos.tolist()]
            n_rand = min(n_random_mask, len(non_bsj_indices))
            if n_rand > 0:
                rand_mask_pos = torch.tensor(non_bsj_indices)[torch.randperm(len(non_bsj_indices))[:n_rand]]
            else:
                rand_mask_pos = torch.tensor([], dtype=torch.long)

            all_mask_pos = torch.cat([bsj_mask_pos, rand_mask_pos]).long()

            for pos in all_mask_pos:
                y_mlm[i, pos] = x[i, pos]
                x_mlm[i, pos] = 4  # MASK token index (adjust if different)

        # Forward pass
        emb = self.model.embedding(x_mlm)
        out, _ = self.model.backbone(emb, mask, None, None)
        y_hat = self.model.mlm_head(out)

        y_hat = y_hat.contiguous().view(-1, y_hat.size(-1))
        y_mlm = y_mlm.contiguous().view(-1)

        loss = self.loss_fn_mlm(y_hat, y_mlm)
        return loss

    def log_epoch_results(self, epoch, losses, times, phase):
        # self.logger.info(f"[{phase.upper()}] Epoch {epoch} Losses: {losses} Times: {times}")
        for key in losses:
            if losses[key] > 0.0:
                key_loss = losses[key]
                key_time = times[key]
                self.logger.info(f"[{phase.upper()}] {key.upper()} | Loss: {key_loss:.4f} | Time: {key_time:.2f}s")
        self.logger.info(f"-" * 50)
            