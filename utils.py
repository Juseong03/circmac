
from typing import Optional, Dict, List, Tuple, Any
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from data import (
    CircRNABindingSitesDataset, CircRNASelfDataset, split_train_valid_test, split_train_valid
)

import os
import random
import json
import gc
import logging
import ast 
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------------------------
# General Utilities
#--------------------------

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"Seeds set to {seed}.")

def get_device(cuda_num: Optional[int] = None) -> str:
    """
    Get the device string for PyTorch.
    """
    if cuda_num is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif cuda_num < 0:
        device = "cpu"
    elif cuda_num in [0, 1, 2, 3]:
        device = f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    return device

def check_max_len(max_len: int, model_name: str) -> int:
    """
    Adjust maximum sequence length based on model constraints.
    Takes the minimum of user-specified max_len and model's position embedding limit.
    This allows fair comparison by passing a smaller max_len (e.g., --max_len 440).
    """
    user_max = max_len - 2  # Account for special tokens

    if model_name.lower() in ['rnabert']:
        model_max = 438
    elif model_name.lower() in ['rnaernie']:
        model_max = 510
    elif model_name.lower() in ['rnafm', 'rnamsm']:
        model_max = 1022
    else:
        logger.info(f'Max length set to {user_max} for model {model_name}')
        return user_max

    result = min(user_max, model_max)
    logger.info(f'Max length set to {result} for model {model_name} (user={user_max}, model_limit={model_max})')
    return result

def count_parameters(module: nn.Module) -> int:
    """
    Count the number of trainable parameters in a module.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def get_optimizer(params: Any, opt_name: str, lr: float = 1e-4, w_decay: Optional[float] = None) -> optim.Optimizer:
    """
    Create an optimizer for model training.
    """
    weight_decay = 0 if w_decay is None else w_decay
    optimizer_dict = {
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
    }
    optimizer_class = optimizer_dict.get(opt_name.lower())
    if optimizer_class is None:
        raise ValueError(f"Optimizer '{opt_name}' not recognized.")
    return optimizer_class(params, lr=lr, weight_decay=weight_decay)

def clean_gpu() -> None:
    """
    Clean GPU memory and perform garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("GPU memory cleaned.")


#--------------------------
# Dataset Preparation
#--------------------------

def prepare_datasets(
    df: pd.DataFrame, 
    df_test: Optional[pd.DataFrame] = None,
    max_len: int = 1024, 
    target: str = 'mirna', 
    seed: int = 42, 
    kmer: int = 1,
    df_extra: Optional[pd.DataFrame] = None
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare datasets for training, validation, and testing.
    """
    if df_test is not None:
        train_df, valid_df = split_train_valid(df, seed=seed, label_column='label')
        test_df = df_test
    else:
        train_df, valid_df, test_df = split_train_valid_test(df, seed=seed, label_column='label')
    train_dataset = CircRNABindingSitesDataset(train_df, max_len=max_len, target_type=target, k=kmer, k_target=kmer)
    valid_dataset = CircRNABindingSitesDataset(valid_df, max_len=max_len, target_type=target, k=kmer, k_target=kmer)
    test_dataset  = CircRNABindingSitesDataset(test_df,  max_len=max_len, target_type=target, k=kmer, k_target=kmer)
    if df_extra is not None:
        extra_dataset = CircRNABindingSitesDataset(df_extra, max_len=max_len, target_type=target, k=kmer, k_target=kmer)
    else:
        extra_dataset = None
    
    return train_dataset, valid_dataset, test_dataset, extra_dataset 


def prepare_self_datasets(df, max_len=1024, seed=42, kmer=1, is_test=False, pair_mode=False):
    if is_test:
        train_df, valid_df, test_df = split_train_valid_test(df, seed=seed)
    else:
        train_df, valid_df = split_train_valid(df, seed=seed)
        test_df = df
    train_dataset = CircRNASelfDataset(train_df, max_len=max_len, k=kmer, pair_mode=pair_mode)
    valid_dataset = CircRNASelfDataset(valid_df, max_len=max_len, k=kmer, pair_mode=pair_mode)
    test_dataset  = CircRNASelfDataset(test_df,  max_len=max_len, k=kmer, pair_mode=pair_mode)
    return train_dataset, valid_dataset, test_dataset 


def save_dataframes(df: pd.DataFrame, path: str = './data/', file_name: str = 'data') -> None:
    """
    Save DataFrame to a JSON file.
    """
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"{file_name}.json")
    df.to_json(save_path, orient='records', lines=True)
    logger.info(f"Data saved to '{save_path}'.")

#--------------------------
# Model Saving and Loading
#--------------------------


def load_component_state_dict(
    model: nn.Module, 
    component_name: str, 
    path: str, 
    device: str = 'cpu', 
    verbose: bool = False
) -> Tuple[nn.Module, bool]:
    try:
        # Check if the file exists.
        if not os.path.exists(path):
            if verbose:
                print(f"X Warning: {component_name} file not found at {path}.")
            return model, False

        # Load state dictionary.
        state_dict = torch.load(path, map_location=device)

        # Get the component from the model.
        component = getattr(model, component_name, None)
        if component is None:
            if verbose:
                print(f"X Warning: Component '{component_name}' not found in the model.")
            return model, False

        # Debugging: Check state_dict compatibility.
        if verbose:
            model_keys = set(component.state_dict().keys())
            state_keys = set(state_dict.keys())
            missing_keys = model_keys - state_keys
            extra_keys = state_keys - model_keys
            if missing_keys:
                print(f"X Missing keys in the loaded state_dict: {missing_keys}")
            if extra_keys:
                print(f"X Extra keys in the loaded state_dict: {extra_keys}")

        # Load the state dict into the component.
        component.load_state_dict(state_dict, strict=False)
        if verbose:
            print(f"O {component_name} successfully loaded from {path}")

        return model, True

    except Exception as e:
        if verbose:
            print(f"X Failed to load {component_name} from {path}. Error: {e}")
        return model, False

def save_component_state_dict(
    component: nn.Module, 
    save_dir: str, 
    name: str, 
    rank: int = 0, 
    verbose: bool = False
) -> bool:
    try:
        # Ensure the save directory exists.
        os.makedirs(save_dir, exist_ok=True)
        component_path = os.path.join(save_dir, f"{name}.pth")

        # Save only if rank is 0.
        if rank == 0:
            torch.save(component.state_dict(), component_path)
            if verbose:
                print(f"O {name} successfully saved to {component_path}")
            return True

        if verbose:
            print(f"X Skipping save for {name} on rank {rank} (not rank 0).")
        return False

    except Exception as e:
        if verbose:
            print(f"X Failed to save {name} to {save_dir}. Error: {e}")
        return False

def save_logs(
    logs: Dict, 
    log_dir: str, 
    log_file: str, 
    verbose: bool = False
) -> None:
    if logs is None or log_dir is None or log_file is None:
        raise ValueError("logs, log_dir and log_file must be provided")
        
    try:
        # Create the full file path and ensure the directory exists.
        file_path = os.path.join(log_dir, f"{log_file}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write logs to file.
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
            
        if verbose:
            print(f"O Training logs saved successfully to {file_path}")
            
    except OSError as e:
        if verbose:
            print(f"X Failed to save logs - OS error: {str(e)}")
        raise
    except Exception as e:
        if verbose:
            print(f"X Failed to save logs - Unexpected error: {str(e)}")
        raise


def save_model(
    model: nn.Module, 
    dir_save: str, 
    model_name: str, 
    experiment_name: str, 
    seed: int, 
    rank: int = 0, 
    pretrain: bool = False, 
    epoch: Optional[int] = None, 
    verbose: bool = False
) -> None:
    # Build the base experiment directory.
    experiment_dir = os.path.join(dir_save, model_name, experiment_name, str(seed))
    subfolder = 'pretrain' if pretrain else 'train'
    save_dir = os.path.join(experiment_dir, subfolder)
    
    if epoch is not None:
        save_dir = os.path.join(save_dir, "epoch", str(epoch))
    
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "model.pth")
    
    torch.save(model.state_dict(), model_save_path)
    if verbose:
        print(f"Model state saved to {model_save_path}")

def save_logs(
    logs: Dict, 
    log_dir: str, 
    log_file: str, 
    verbose: bool = False
) -> None:
    if logs is None or log_dir is None or log_file is None:
        raise ValueError("logs, log_dir and log_file must be provided")
        
    try:
        file_path = os.path.join(log_dir, f"{log_file}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)
        if verbose:
            print(f"Training logs saved successfully to {file_path}")
    except Exception as e:
        if verbose:
            print(f"Failed to save logs: {e}")
        raise

def load_model(
        model, 
        dir_save, 
        model_name='mamba', 
        experiment_name='default', 
        seed=42, 
        device='cpu', 
        pretrain=False, 
        load_pretrain_name=None, 
        epoch=None, 
        verbose=False
    ):
    """
    Load the model's state dictionary for pretraining or training.
    """
    try:
        # Define experiment directory
        exp_name = experiment_name if load_pretrain_name is None else load_pretrain_name
        load_dir = os.path.join(dir_save, model_name, exp_name, str(seed))
        
        # Determine the subdirectory based on the pretrain flag.
        if pretrain:
            load_dir = os.path.join(load_dir, 'pretrain')
        else:
            load_dir = os.path.join(load_dir, 'train')
        
        # Append the epoch subdirectory if an epoch is provided.
        if epoch is not None:
            load_dir = os.path.join(load_dir, "epoch", str(epoch))

        model_path = os.path.join(load_dir, "model.pth")

        # Load main model state dict
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            if verbose:
                print(f"O Model state loaded successfully from {model_path}")
        else:
            if verbose:
                print(f"! Warning: Model state file not found at {model_path}.")
            return model
        # Component paths
        component_paths = {
            "embedding": os.path.join(load_dir, "embedding.pth"),
            "backbone": os.path.join(load_dir, "backbone.pth"),
        }

        if pretrain:
            component_paths.update({
                "token_dropout": os.path.join(load_dir, "token_dropout.pth"),
                "mlm_head": os.path.join(load_dir, "mlm_head.pth"),
                "ntp_head": os.path.join(load_dir, "ntp_head.pth"),
                "proj_contrastive": os.path.join(load_dir, "proj_contrastive.pth"),
            })
        else:
            component_paths.update({
                "proj_target": os.path.join(load_dir, "proj_target.pth"),
                "convblock": os.path.join(load_dir, "convblock.pth"),
                "binding_head": os.path.join(load_dir, "binding_head.pth"),
                "binding_site_head": os.path.join(load_dir, "binding_site_head.pth"),
            })

        # Load each component
        for component, path in component_paths.items():
            model, success = load_component_state_dict(model, component, path, device=device, verbose=verbose)
            if not success and verbose:
                print(f"X Failed to load {component} from {path}. Skipping.")

        # Move model to device
        model.to(device)
        if verbose:
            print(f"O Model successfully moved to {device}")

        return model

    except Exception as e:
        if verbose:
            print(f"X Error while loading the model: {e}")
        return model

#--------------------------
# Metrics and Loss Functions
#--------------------------
def calculate_class_weights_from_df(df: pd.DataFrame, 
                                  class_col_name: str = 'sites', 
                                  length_col_name: str = 'length',
                                  device: str = 'cpu') -> torch.Tensor:
    num_total_pos = 0
    num_total_neg = 0

    for idx, row in df.iterrows():
        class_data = row[class_col_name]
        try:
            actual_length = int(row[length_col_name])
        except (ValueError, TypeError):
            actual_length = len(class_data) if isinstance(class_data, (list, np.ndarray)) else 0

        if actual_length == 0:
            continue

        class_vector_for_sample: list
        if isinstance(class_data, str):
            try:
                class_vector_for_sample = ast.literal_eval(class_data)
                if not isinstance(class_vector_for_sample, list):
                    print(f"Row {idx}: Parsed '{class_col_name}' string did not result in a list. Skipping for weight calculation.")
                    continue
            except (ValueError, SyntaxError):
                print(f"Row {idx}: Could not parse '{class_col_name}' string '{class_data}'. Skipping for weight calculation.")
                continue
        elif isinstance(class_data, (list, np.ndarray)):
            class_vector_for_sample = list(class_data)
        else:
            print(f"Row {idx}: Unexpected '{class_col_name}' format: {type(class_data)}. Skipping for weight calculation.")
            continue
        
        if len(class_vector_for_sample) < actual_length:
            print(f"Row {idx}: Site vector length {len(class_vector_for_sample)} is less than actual_length {actual_length}. Using shorter length.")

            current_valid_labels = class_vector_for_sample 
        else:
            current_valid_labels = class_vector_for_sample[:actual_length]

        pos_in_sample = sum(s == 1 or s == 1.0 for s in current_valid_labels)
        neg_in_sample = sum(s == 0 or s == 0.0 for s in current_valid_labels)
        
        num_total_pos += pos_in_sample
        num_total_neg += neg_in_sample

    if num_total_pos == 0 or num_total_neg == 0:
        print("One of the classes (site/non-site) has zero samples across the dataset. Using equal weights [1.0, 1.0].")
        return torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)

    weight_for_neg = 1.0
    weight_for_pos = float(num_total_neg) / num_total_pos

    logger.info(f"Calculated class weights for sites - Non-site: {weight_for_neg:.4f}, Site: {weight_for_pos:.4f}")
    logger.info(f"Total valid sites: {num_total_pos}, Total valid non-sites: {num_total_neg}")

    return torch.tensor([weight_for_neg, weight_for_pos], dtype=torch.float32).to(device)


def without_pads(
    preds: torch.Tensor, 
    labels: torch.Tensor, 
    len_circ: torch.Tensor, 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove padding tokens based on sequence lengths.
    """
    device = preds.device
    if not (len(preds) == len(labels) == len(len_circ)):
        raise ValueError("Preds, labels, and len_circ must have the same length.")

    preds_wo_pads, labels_wo_pads = [], []
    for i, n_pad in enumerate(len_circ):
        n = int(round(n_pad.item()))
        preds_wo_pads.append(preds[i, :n] if n > 0 else preds[i, :])
        labels_wo_pads.append(labels[i, :n] if n > 0 else labels[i, :])
    return torch.cat(preds_wo_pads).to(device), torch.cat(labels_wo_pads).to(device)


from typing import Tuple, List

def without_pads_sites(
    preds: torch.Tensor, 
    labels: torch.Tensor, 
    len_circ: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Remove padding tokens based on sequence lengths.
    Returns:
        flat_preds: [total_tokens, D]
        flat_labels: [total_tokens]
        split_lengths: list of sequence lengths (int)
    """
    device = preds.device
    if not (len(preds) == len(labels) == len(len_circ)):
        raise ValueError("Preds, labels, and len_circ must have the same length.")

    preds_wo_pads, labels_wo_pads = [], []
    split_lengths = []

    for i, n_pad in enumerate(len_circ):
        n = int(round(n_pad.item()))
        if n > 0:
            preds_wo_pads.append(preds[i, :n])
            labels_wo_pads.append(labels[i, :n])
            split_lengths.append(n)
        else:
            preds_wo_pads.append(preds[i, :])
            labels_wo_pads.append(labels[i, :])
            split_lengths.append(preds[i, :].shape[0])

    flat_preds = torch.cat(preds_wo_pads).to(device)
    flat_labels = torch.cat(labels_wo_pads).to(device)
    return flat_preds, flat_labels, split_lengths


def cal_loss_sites(
    loss_fn: nn.Module,
    preds: torch.Tensor,
    labels: torch.Tensor,
    len_circ: torch.Tensor
) -> torch.Tensor:
    preds, labels = without_pads(preds, labels, len_circ)
    return loss_fn(preds, labels.long())
def calculate_metrics(
    preds_np: np.ndarray, 
    labels_np: np.ndarray, 
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    """
    acc = accuracy_score(labels_np, preds_np)
    f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
    precision_macro = precision_score(labels_np, preds_np, average='macro', zero_division=0)
    recall_macro = recall_score(labels_np, preds_np, average='macro', zero_division=0)
    mcc = matthews_corrcoef(labels_np, preds_np) if len(set(labels_np)) > 1 else 0.0
    try:
        if preds_np.ndim > 1 and preds_np.shape[1] > 1:
            auc = roc_auc_score(labels_np, preds_np, multi_class="ovo", average="weighted")
        else:
            auc = roc_auc_score(labels_np, preds_np) if len(set(labels_np)) > 1 else 0.0
    except ValueError:
        auc = 0.0

    return {
        'acc': acc,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'mcc': mcc,
        'auc': auc
    }

def find_best_threshold_site(labels_np, probs_np, metric='f1_pos', thresholds=np.linspace(0.1, 0.9, 17)):
    best_score, best_t = 0, 0.5
    for t in thresholds:
        m = calculate_site_metrics(labels_np, probs_np, threshold=t)
        if m[metric] > best_score:
            best_score, best_t = m[metric], t
    return best_t, best_score

    
def cal_score_sites(
    preds_logits: torch.Tensor, 
    labels: torch.Tensor,       
    len_circ: torch.Tensor,     
    threshold: float = 0.5,
    threshold_sweep: bool = True,            # ✅ 추가: sweep 여부
    sweep_metric: str = 'f1_macro',             # ✅ 기준 metric
    sweep_range=np.linspace(0.1, 0.9, 9)      # ✅ sweep 범위
) -> Dict[str, float]:

    flat_logits, flat_labels, split_lengths = without_pads_sites(preds_logits, labels, len_circ)

    if flat_labels.numel() == 0:
        return {
            'acc': 0.0, 'f1_macro': 0.0, 'f1_pos': 0.0,
            'prec_macro': 0.0, 'prec_pos': 0.0,
            'rec_macro': 0.0, 'rec_pos': 0.0,
            'mcc': 0.0, 'roc_auc': 0.5, 'auprc': 0.0,
            'span_precision': 0.0, 'span_recall': 0.0, 'span_f1': 0.0,
            'TP': 0, 'FP': 0, 'FN': 0
        }

    labels_np = flat_labels.cpu().numpy()

    # === 확률 변환 ===
    if preds_logits.dim() == 3 and preds_logits.size(-1) == 2:
        probs_positive_class_np = torch.softmax(flat_logits, dim=-1)[:, 1].cpu().numpy()
    elif preds_logits.dim() == 2 or (preds_logits.dim() == 3 and preds_logits.size(-1) == 1):
        if flat_logits.dim() == 2 and flat_logits.shape[1] == 1:
            flat_logits = flat_logits.squeeze(-1)
        probs_positive_class_np = torch.sigmoid(flat_logits).cpu().numpy()
    else:
        raise ValueError(f"Unsupported preds_logits shape: {preds_logits.shape}")

    # === threshold sweep ===
    if threshold_sweep:
        best_t, best_score = find_best_threshold_site(labels_np, probs_positive_class_np, metric=sweep_metric, thresholds=sweep_range)
        threshold = best_t
    else:
        best_t = threshold

    # === binary prediction ===
    preds_binary_np = (probs_positive_class_np >= threshold).astype(int)

    # === span용 배치 split ===
    labels_batch, preds_batch = [], []
    idx = 0
    for l in split_lengths:
        labels_batch.append(labels_np[idx:idx+l].tolist())
        preds_batch.append(preds_binary_np[idx:idx+l].tolist())
        idx += l

    # === 계산 ===
    base_metrics = calculate_site_metrics(labels_np, probs_positive_class_np, threshold)
    span_metrics = span_f1(preds_batch, labels_batch, strict=True)
    base_metrics.update(span_metrics)

    # ✅ threshold 반환 추가
    base_metrics["threshold_used"] = threshold
    if threshold_sweep:
        base_metrics[f"best_{sweep_metric}"] = base_metrics[sweep_metric]

    return base_metrics


def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    probs = torch.softmax(logits, dim=-1)
    pt = probs[torch.arange(len(labels)), labels]
    focal_weight = (1 - pt) ** gamma
    alpha_weight = alpha * (labels == 1).float() + (1 - alpha) * (labels == 0).float()
    loss = focal_weight * alpha_weight * ce_loss
    return loss.mean()

def cal_loss_sites_focal(
    pred_sites,    # (batch, seq_len, 2)
    label_sites,   # (batch, seq_len)
    lengths,       # (batch,)
    sample_ratio=0.1,
    alpha=0.25,
    gamma=2.0
):
    total_loss = 0.0
    batch_size = pred_sites.size(0)

    for i in range(batch_size):
        # 안전하게 flatten
        if lengths.dim() == 2 and lengths.size(1) == 1:
            lengths = lengths.view(-1)
        
        # 이후 loss loop 내부에서
        length = int(lengths[i].item())
        logits = pred_sites[i, :length]
        labels = label_sites[i, :length]
        
        # Pos/Neg 인덱스 구하기
        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

        # Negative 샘플링
        k = int(len(neg_idx) * sample_ratio)
        if k > 0:
            sampled_neg_idx = neg_idx[torch.randperm(len(neg_idx))[:k]]
            selected_idx = torch.cat([pos_idx, sampled_neg_idx])
        else:
            selected_idx = pos_idx  # 거의 없을 경우 fallback

        if len(selected_idx) == 0:
            continue  # skip empty sample

        selected_logits = logits[selected_idx]
        selected_labels = labels[selected_idx]
        loss = focal_loss(selected_logits, selected_labels, alpha=alpha, gamma=gamma)
        total_loss += loss

    return total_loss / batch_size

def batched_focal_loss(
    logits,             # (B, L, 2)  ← 이미 [CLS] 제거된 상태
    labels,             # (B, L)     ← logits와 동일한 길이
    lengths,            # (B,)
    class_weights=None,
    gamma=2.0,
    ignore_index=-100,
    sample_ratio=0.1
):
    B, L, C = logits.shape           # 한 번만 정의
    device = logits.device

    # 1. Flatten
    logits_flat = logits.reshape(-1, C)     # (B*L, 2)
    labels_flat = labels.reshape(-1)        # (B*L,)

    # 2. Valid mask: length mask & ignore index
    range_tensor = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
    lengths = lengths.view(-1, 1)                               # (B, 1)
    length_mask = range_tensor < lengths                        # (B, L)
    ignore_mask = labels != ignore_index                       # (B, L)
    valid_mask = (length_mask & ignore_mask).reshape(-1)       # (B*L,)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_valid = logits_flat[valid_mask]
    labels_valid = labels_flat[valid_mask].long()

    # 3. Focal weight
    ce_loss = F.cross_entropy(logits_valid, labels_valid, reduction='none')  # (N,)
    probs = torch.softmax(logits_valid, dim=1)
    pt = probs[torch.arange(len(labels_valid), device=device), labels_valid]
    focal_weight = (1 - pt) ** gamma

    # 4. Class weight (α)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        alpha = class_weights[labels_valid]
    else:
        alpha = torch.ones_like(focal_weight)

    loss = alpha * focal_weight * ce_loss

    # 5. Positional sampling (if enabled)
    if sample_ratio < 1.0:
        pos_idx = (labels_valid == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels_valid == 0).nonzero(as_tuple=True)[0]

        num_neg = int(len(neg_idx) * sample_ratio)
        if num_neg > 0:
            sampled_neg = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]
            selected_idx = torch.cat([pos_idx, sampled_neg])
            loss = loss[selected_idx]

    return loss.mean()
#--------------------------
# Pretraining Utilities
#--------------------------

def create_mlm_inputs_and_labels(inputs, mask_ratio=0.15, device='cpu', mask_token_id=4, attention_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(inputs, torch.Tensor):
        raise ValueError("Inputs must be a torch.Tensor.")
    inputs = inputs.clone().to(device)
    labels = inputs.clone()
    mask = torch.bernoulli(torch.full(labels.shape, mask_ratio)).bool().to(device)
    if attention_mask is not None:
        mask = mask & attention_mask.bool().to(device)  # PAD 위치 마스킹 제외
    inputs[mask] = mask_token_id
    labels[~mask] = -100  # CrossEntropyLoss 무시
    return inputs, labels


def create_ntp_inputs_and_labels(
    seqs: torch.Tensor, 
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create inputs and labels for Next Token Prediction (NTP).
    
    Args:
        seqs: Input tensor of token sequences (batch_size, seq_len)
        device: Target device for the output tensors
        
    Returns:
        Tuple of (inputs_ntp, labels_ntp)
        
    Raises:
        ValueError: If seqs is not a torch.Tensor
    """
    if not isinstance(seqs, torch.Tensor):
        raise ValueError("Input sequences must be a torch.Tensor.")
    
    seqs = seqs.to(device)
    inputs_ntp = seqs[:, :-1]  # Remove last token for inputs
    labels_ntp = seqs[:, 1:]  # Remove first token for labels

    # Add padding to align dimensions
    padding = torch.full((inputs_ntp.size(0), 1), fill_value=0, dtype=torch.long).to(device)
    inputs_ntp = torch.cat((padding, inputs_ntp), dim=1)
    labels_ntp = F.pad(labels_ntp, (0, 1), value=-100)  # -100 for ignored positions
    
    return inputs_ntp, labels_ntp


def create_ntp_inputs_and_labels_with_mask(
    seqs: torch.Tensor, 
    mask: torch.Tensor, 
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(seqs, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("Input sequences and mask must be torch.Tensors.")
    
    seqs = seqs.to(device)
    mask = mask.to(device)
    
    inputs_ntp = seqs[:, :-1]
    labels_ntp = seqs[:, 1:]
    mask_ntp = mask[:, 1:]

    padding = torch.full((inputs_ntp.size(0), 1), fill_value=0, dtype=torch.long).to(device)
    inputs_ntp = torch.cat((padding, inputs_ntp), dim=1)
    labels_ntp = F.pad(labels_ntp, (0, 1), value=-100)
    mask_ntp = F.pad(mask_ntp, (0, 1), value=0)

    labels_ntp = labels_ntp.masked_fill(mask_ntp == 0, -100)
    
    return inputs_ntp, labels_ntp


def contrastive_loss(
    embeddings1: torch.Tensor, 
    embeddings2: torch.Tensor, 
    labels: torch.Tensor, 
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Compute contrastive loss with temperature scaling.
    
    Args:
        embeddings1: Embeddings of the first set (batch_size, embed_dim)
        embeddings2: Embeddings of the second set (batch_size, embed_dim)
        labels: Binary labels indicating positive pairs (1) and negatives (0)
        temperature: Temperature scaling parameter
        
    Returns:
        Contrastive loss value
    """
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # Compute cosine similarity matrix
    cosine_sim = torch.mm(embeddings1, embeddings2.T) / temperature

    # Expand labels to match the similarity matrix shape
    labels = labels.unsqueeze(1).expand_as(cosine_sim)

    # Positive and negative similarities
    positive_pairs = cosine_sim[labels == 1]
    negative_pairs = cosine_sim[labels == 0]

    # Compute positive and negative losses
    positive_loss = -torch.log(torch.exp(positive_pairs).sum() / torch.exp(cosine_sim).sum())
    negative_loss = -torch.log(1 - torch.exp(negative_pairs).sum() / torch.exp(cosine_sim).sum())

    return (positive_loss + negative_loss) / 2


def masked_contrastive_loss(
    embeddings1: torch.Tensor, 
    embeddings2: torch.Tensor, 
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Compute the normalized temperature-scaled cross-entropy loss (NT-Xent loss).
    
    Args:
        embeddings1: Embeddings of the original set (batch_size, embed_dim)
        embeddings2: Embeddings of the augmented or paired set (batch_size, embed_dim)
        temperature: Temperature scaling parameter
        
    Returns:
        Contrastive loss value
    """
    # Normalize embeddings to unit vectors
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)

    # Compute similarity scores
    similarity_matrix = torch.mm(embeddings1, embeddings2.t())  # Shape: (batch_size, batch_size)

    # Scale by temperature
    similarity_matrix /= temperature

    # Create the mask to exclude self-similarity
    batch_size = embeddings1.size(0)
    mask = torch.eye(batch_size, device=embeddings1.device).bool()

    # Apply mask and compute the logits
    positive_logits = similarity_matrix[mask].view(batch_size, -1)  # Positive pairs
    negative_logits = similarity_matrix[~mask].view(batch_size, -1)  # Negative pairs

    # Construct labels for cross-entropy loss
    labels = torch.arange(batch_size, device=embeddings1.device)

    # Combine logits for contrastive learning
    logits = torch.cat([positive_logits, negative_logits], dim=1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss

def cal_score_self(
    preds_logits: torch.Tensor,
    labels: torch.Tensor,
    task_type: str,  # 'binary', 'multiclass', 'matrix-binary'
    threshold: float = 0.5,
    auto_threshold: bool = False,
    metric: str = 'f1'
) -> Dict[str, float]:
    """
    Self-supervised task 평가 지표 계산 함수.
    `auto_threshold=True`일 경우, 지정된 metric 기준으로 binary task에 대해 최적 threshold를 자동 탐색함.
    """
    if task_type == 'binary':
        probs = torch.sigmoid(preds_logits.detach().view(-1)).cpu().numpy()
        labels_np = labels.detach().view(-1).long().cpu().numpy()

        if auto_threshold:
            threshold = find_best_threshold(labels_np, probs, metric)

        preds_bin = (probs >= threshold).astype(int)
        return calculate_binary_classification_metrics(labels_np, probs, preds_bin)

    elif task_type == 'multiclass':
        preds_flat = preds_logits.detach().view(-1, preds_logits.size(-1)).cpu()
        labels_flat = labels.detach().view(-1).cpu().numpy()

        preds_probs = torch.softmax(preds_flat, dim=-1).cpu().numpy()
        preds_classes = preds_probs.argmax(axis=-1)

        return calculate_multiclass_metrics(labels_flat, preds_probs, preds_classes)

    elif task_type == 'matrix-binary':
        probs = torch.sigmoid(preds_logits.detach()).cpu().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().reshape(-1)

        if auto_threshold:
            threshold = find_best_threshold(labels_np, probs, metric)

        preds_bin = (probs >= threshold).astype(int)
        return calculate_binary_classification_metrics(labels_np, probs, preds_bin)

    else:
        raise ValueError(f"Unknown task type: {task_type}")

def find_best_threshold(
    labels: np.ndarray, 
    probs: np.ndarray, 
    metric: str = 'f1', 
    step: float = 0.01
) -> float:
    thresholds = np.arange(0.0, 1.0 + step, step)
    best_score = -1.0
    best_threshold = 0.5

    for th in thresholds:
        preds_bin = (probs >= th).astype(int)
        try:
            if metric == 'f1':
                score = f1_score(labels, preds_bin)
            elif metric == 'mcc':
                score = matthews_corrcoef(labels, preds_bin)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        except:
            continue  # avoid exception due to undefined metric

        if score > best_score:
            best_score = score
            best_threshold = th

    return best_threshold

def calculate_binary_classification_metrics(labels, probs, preds_bin):
    return {
        'acc': accuracy_score(labels, preds_bin),
        'f1': f1_score(labels, preds_bin),
        'precision': precision_score(labels, preds_bin),
        'recall': recall_score(labels, preds_bin),
    }

def calculate_multiclass_metrics(labels, probs, preds):
    return {
        'acc': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro')
    }

def calculate_binary_matrix_metrics(labels, probs, preds_bin):
    flat_labels = labels.reshape(-1)
    flat_preds = preds_bin.reshape(-1)
    flat_probs = probs.reshape(-1)
    return calculate_binary_classification_metrics(flat_labels, flat_probs, flat_preds)


import torch
import torch.nn as nn

class UncertaintyWeightingLoss(nn.Module):
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        # log(sigma^2) 형태로 초기화 (학습 가능한 파라미터)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(())) for task in task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        loss_dict = {}
        for task in self.task_names:
            log_var = self.log_vars[task]
            precision = torch.exp(-log_var)
            weighted_loss = precision * losses[task] + log_var
            loss_dict[task] = weighted_loss.item()
            total_loss += weighted_loss
        return total_loss, loss_dict
