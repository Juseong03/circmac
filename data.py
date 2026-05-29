import ast
from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict
from transformers import T5Tokenizer
from multimolecule import RnaTokenizer
from sklearn.model_selection import train_test_split
import random
import itertools
import numpy as np
from multiprocessing import Pool
import os
import pickle
import ast
import logging
from typing import Any, Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------------------------------------------------
# Tokenizer Classes
#--------------------------------------------------
def generate_kmer_vocab(k: int) -> Dict[str, int]:
    bases = ['A', 'C', 'G', 'U']
    kmer_list = [''.join(p) for p in itertools.product(bases, repeat=k)]
    vocab = {kmer: idx for idx, kmer in enumerate(kmer_list, start=1)}
    vocab["[PAD]"] = 0
    return vocab


class KmerTokenizer:
    def __init__(self, k: int):
        self.k = k
        self.base_vocab = OrderedDict([
            ('<pad>', 0), ('<cls>', 1), ('<eos>', 2), ('<unk>', 3),
            ('<mask>', 4), ('<null>', 5), ('A', 6), ('C', 7),
            ('G', 8), ('U', 9), ('N', 10)
        ])
        self.vocab = self._generate_kmer_vocab(k)
        self.vocab_size = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _generate_kmer_vocab(self, k: int) -> Dict[str, int]:
        bases = [key for key in self.base_vocab.keys() if len(key) == 1 and key.isalpha()]
        kmer_list = [''.join(p) for p in itertools.product(bases, repeat=k)]
        kmer_vocab = {kmer: idx + len(self.base_vocab) for idx, kmer in enumerate(kmer_list)}
        kmer_vocab.update({token: self.base_vocab[token] for token in self.base_vocab})
        return kmer_vocab

    def tokenize(self, sequence: str) -> List[str]:
        if not sequence or len(sequence) < self.k:
            return ["<pad>"]
        return [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]

    def encode(self, sequence: str, max_len: int) -> Dict[str, torch.Tensor]:
        tokens = self.tokenize(sequence)
        token_ids = [self.base_vocab['<cls>']] + \
                    [self.vocab.get(token, self.base_vocab['<unk>']) for token in tokens] + \
                    [self.base_vocab['<eos>']]

        # adjust length
        if len(token_ids) < max_len:
            attention_mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
            token_ids += [self.base_vocab['<pad>']] * (max_len - len(token_ids))
        else:
            attention_mask = [1] * max_len
            token_ids = token_ids[:max_len]

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }



#--------------------------------------------------
# Dataset Base Class
#--------------------------------------------------

class BaseCircRNADataset(Dataset):
    def __init__(self, df: Any, max_len: int = 512, padding_value: int = -100, 
                 target_type: str = 'mirna', k: int = 1, k_target: int = 1, 
                 site_expansion_window: int = 0):
        self.df = df
        self.max_len_circrna = max_len
        self.padding_value = padding_value
        self.target_type = target_type.lower()
        self.k = k
        self.k_target = k_target
        self.site_expansion_window = site_expansion_window # <-- 저장

        if k > 1:
            self.circrna_tokenizer = KmerTokenizer(k)
        else:
            try:
                self.circrna_tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnabert')
            except Exception as e:
                logger.error(f"Error loading RnaTokenizer for circRNA: {e}. Falling back to KmerTokenizer with k=1")
                self.circrna_tokenizer = KmerTokenizer(1)

        if self.target_type in ['mirna', 'mirnas', 'micro']:
            self.max_len_target = 25
            if k_target > 1:
                self.target_tokenizer = KmerTokenizer(k_target)
            else:
                try:
                    self.target_tokenizer = RnaTokenizer.from_pretrained('multimolecule/rnabert')
                except Exception as e:
                    logger.error(f"Error loading RnaTokenizer for target: {e}. Falling back to KmerTokenizer with k=1")
                    self.target_tokenizer = KmerTokenizer(1)
        elif self.target_type in ['protein', 'rbp', 'proteins']:
            self.max_len_target = 512
            try:
                self.target_tokenizer = T5Tokenizer.from_pretrained(
                    'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False
                )
            except Exception as e:
                logger.error(f"Error loading protein tokenizer: {e}. Using simple tokenizer.")
                self.target_tokenizer = KmerTokenizer(1)
        else:
            raise ValueError(f"Target type '{target_type}' not recognized. Use 'mirna' or 'protein'/'rbp'.")

        self.vocab_size = getattr(self.circrna_tokenizer, 'vocab_size', None)
        if self.vocab_size is None and hasattr(self.circrna_tokenizer, 'vocab'):
            self.vocab_size = len(self.circrna_tokenizer.vocab)

    def __len__(self) -> int:
        return len(self.df)

    def _reverse_complement(self, sequence: str) -> str:
        if not sequence:
            return ""
        complement_map = str.maketrans("ACGUNT", "UGCAAN")
        return sequence.translate(complement_map)[::-1]

    def _tokenize_sequence(self, tokenizer: Any, sequence: str, max_len: int, k: int = 1) -> Dict[str, torch.Tensor]:
        if not sequence:
            pad_token_id = (tokenizer.base_vocab['<pad>'] 
                            if hasattr(tokenizer, 'base_vocab') and '<pad>' in tokenizer.base_vocab 
                            else getattr(tokenizer, 'pad_token_id', 0))
            return {
                'input_ids': torch.full((max_len,), pad_token_id, dtype=torch.long),
                'attention_mask': torch.zeros(max_len, dtype=torch.long)
            }
        
        if k > 1 or isinstance(tokenizer, KmerTokenizer):
            return tokenizer.encode(sequence, max_len)
        else:
            try:
                tokens = tokenizer(
                    sequence,
                    padding='max_length',
                    truncation=True,
                    max_length=max_len,
                    return_tensors='pt'
                )
                return {
                    'input_ids': tokens['input_ids'].squeeze(),
                    'attention_mask': tokens['attention_mask'].squeeze()
                }
            except Exception as e:
                logger.error(f"Error using HuggingFace tokenizer: {e}. Falling back to simple tokenization.")
                simple_tokenizer = KmerTokenizer(1)
                return simple_tokenizer.encode(sequence, max_len)

    def _prepare_sites_tensor(self, sequence: str, sites: Any, k: int) -> torch.Tensor:
        seq_len = len(sequence)
        max_kmer_len = seq_len - k + 1 if seq_len >= k else 0

        # Step 1: k-mer 단위로 site label 구성
        kmer_labels = []
        for i in range(max_kmer_len):
            window = sites[i:i + k]
            cleaned = [s for s in window if s is not None]
            label = max(cleaned, default=0)
            kmer_labels.append(label)

        # Step 2: Special tokens (<cls>, <eos>)
        kmer_labels = [self.padding_value] + kmer_labels + [self.padding_value]  # <cls> and <eos>

        # Step 3: Padding to max_len
        if len(kmer_labels) < self.max_len_circrna:
            pad_len = self.max_len_circrna - len(kmer_labels)
            kmer_labels += [self.padding_value] * pad_len
        else:
            kmer_labels = kmer_labels[:self.max_len_circrna]

        return torch.tensor(kmer_labels, dtype=torch.long)

    def _get_sequence_data(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        circrna_seq = str(row.get('circRNA', ""))
        circrna_length = int(row.get('length', len(circrna_seq)))
        target_seq = ""
        if self.target_type in ['mirna', 'mirnas', 'micro']:
            for col in ['miRNA', 'mirna', 'target', 'micro_rna']:
                if col in row and row[col]:
                    target_seq = str(row[col])
                    break
        else:
            for col in ['RBP', 'rbp', 'protein', 'target']:
                if col in row and row[col]:
                    target_seq = str(row[col])
                    break
        binding = float(row.get('binding', 0))
        sites = row.get('sites', [])

        if isinstance(sites, str):
            try:
                sites = ast.literal_eval(sites)
            except Exception as e:
                logger.error(f"Error parsing sites string in _get_sequence_data: {e}. Using empty list instead.")
                sites = []
        return {
            'circrna_seq': circrna_seq,
            'target_seq': target_seq,
            'circrna_length': circrna_length,
            'binding': binding,
            'sites': sites
        }



class CircRNABindingSitesDataset(BaseCircRNADataset):
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self._get_sequence_data(idx)

        circrna_seq = data['circrna_seq']
        target_seq = data['target_seq']
        circrna_length = data['circrna_length']
        binding = data['binding']
        sites = data['sites']

        circrna_tokens = self._tokenize_sequence(self.circrna_tokenizer, circrna_seq, self.max_len_circrna, self.k)
        circrna_rc_seq = self._reverse_complement(circrna_seq)
        circrna_rc_tokens = self._tokenize_sequence(self.circrna_tokenizer, circrna_rc_seq, self.max_len_circrna, self.k)
        target_tokens = self._tokenize_sequence(self.target_tokenizer, target_seq, self.max_len_target, self.k_target)

        
        sites_tensor = self._prepare_sites_tensor(circrna_seq, sites, self.k)

        return {
            'circRNA': circrna_tokens['input_ids'],
            'circRNA_mask': circrna_tokens['attention_mask'],
            'circRNA_rc': circrna_rc_tokens['input_ids'],
            'circRNA_rc_mask': circrna_rc_tokens['attention_mask'],
            'target': target_tokens['input_ids'],
            'target_mask': target_tokens['attention_mask'],
            'binding': torch.tensor([binding], dtype=torch.float32),
            'sites': sites_tensor,
            'length': torch.tensor([circrna_length], dtype=torch.float32)
        }


class CircRNASelfDataset(BaseCircRNADataset):
    def __init__(
            self,
            df: Any,
            max_len: int = 512,
            padding_value: int = -100,
            target_type: str = 'mirna',
            k: int = 1,
            k_ss: Optional[int] = None,
            pair_mode: bool = False,
        ):

        super().__init__(df, max_len, padding_value, target_type, k)
        self.k_ss = k_ss if k_ss is not None else self.k
        self.ss_vocab = self._generate_ss_vocab(self.k_ss)
        self.ss_vocab_size = len(self.ss_vocab)
        self.ss_inverse_vocab = {v: k for k, v in self.ss_vocab.items()}

        self.pair_mode = pair_mode
        if pair_mode:
            self._build_pair_groups()

    def _build_pair_groups(self):
        """Group records by circRNA_id for SS-pair contrastive learning.
        Each group contains indices of the same sequence with different SS predictions.
        """
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx in range(len(self.df)):
            cid = self.df.iloc[idx].get('circRNA_id', str(idx))
            groups[cid].append(idx)
        self.pair_groups = list(groups.values())  # List[List[int]]

    def __len__(self) -> int:
        if self.pair_mode:
            return len(self.pair_groups)
        return len(self.df)

    def _get_single_item(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        seq = str(row.get("circRNA", ""))
        structure = str(row.get("structure", "." * len(seq)))[:len(seq)]
        length = int(row.get("length", len(seq)))

        ss_labels = self._safe_parse_list(row.get("ss_labels", []))
        ss_labels_multi = row.get("ss_labels_multi", [])

        seq_tokens = self._tokenize_sequence(self.circrna_tokenizer, seq, self.max_len_circrna, self.k)
        seq_rc = self._reverse_complement(seq)
        seq_rc_tokens = self._tokenize_sequence(self.circrna_tokenizer, seq_rc, self.max_len_circrna, self.k)

        ss_tokens = self._encode_ss(structure, self.max_len_circrna)

        pairing = self._dotbracket_to_pairing_matrix(structure)
        pairing = F.pad(pairing, (0, self.max_len_circrna - pairing.shape[1], 0, self.max_len_circrna - pairing.shape[0]))
        pairing = pairing[:self.max_len_circrna, :self.max_len_circrna]

        mask = seq_tokens["attention_mask"].unsqueeze(1)
        pairing_masked = pairing * mask * mask.transpose(0, 1)

        return {
            "circRNA": seq_tokens["input_ids"],
            "circRNA_mask": seq_tokens["attention_mask"],
            "circRNA_rc": seq_rc_tokens["input_ids"],
            "circRNA_rc_mask": seq_rc_tokens["attention_mask"],
            "structure": ss_tokens["input_ids"],
            "length": torch.tensor(length, dtype=torch.float32),
            "pairing": pairing,
            "pairing_masked": pairing_masked,
            "ss_labels": self._pad_or_truncate_label(ss_labels, self.max_len_circrna),
            "ss_labels_multi": self._pad_or_truncate_label(ss_labels_multi, self.max_len_circrna)
        }

    def _generate_ss_vocab(self, k: int) -> Dict[str, int]:
        bases = ['(', ')', '.']
        kmer_list = [''.join(p) for p in itertools.product(bases, repeat=k)]
        vocab = {token: idx for idx, token in enumerate(kmer_list, start=1)}
        vocab["[PAD]"] = 0
        return vocab

    def _tokenize_ss(self, sequence: str) -> List[str]:
        if not sequence or len(sequence) < self.k_ss:
            return ["[PAD]"]
        return [sequence[i:i+self.k_ss] for i in range(len(sequence) - self.k_ss + 1)]

    def _encode_ss(self, sequence: str, max_len: int) -> Dict[str, torch.Tensor]:
        tokens = self._tokenize_ss(sequence)
        token_ids = [self.ss_vocab.get(token, self.ss_vocab["[PAD]"]) for token in tokens]
        if len(token_ids) < max_len:
            attention_mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
            token_ids += [self.ss_vocab["[PAD]"]] * (max_len - len(token_ids))
        else:
            attention_mask = [1] * max_len
            token_ids = token_ids[:max_len]
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

    def _dotbracket_to_pairing_matrix(self, dotbracket: str) -> torch.Tensor:
        L = len(dotbracket)
        doubled = dotbracket + dotbracket
        stack = []
        mat = torch.zeros((L, L), dtype=torch.float32)
        for i, char in enumerate(doubled):
            pos = i % L
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    a, b = j % L, i % L
                    if a != b:
                        mat[a, b] = 1.0
                        mat[b, a] = 1.0
        return mat

    def _pad_or_truncate_label(self, label: List[int], max_len: int) -> torch.Tensor:
        label = [self.padding_value] + label + [self.padding_value]  # <cls> and <eos>
        if len(label) < max_len:
            label += [self.padding_value] * (max_len - len(label))
        else:
            label = label[:max_len]
        return torch.tensor(label, dtype=torch.long)

    def _safe_parse_list(self, val):
        if isinstance(val, list):
            return val
        try:
            return ast.literal_eval(val)
        except Exception:
            return []

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.pair_mode:
            group = self.pair_groups[idx]
            idx1 = group[0]
            idx2 = group[random.randint(1, len(group) - 1)] if len(group) > 1 else group[0]
            item = self._get_single_item(idx1)
            item2 = self._get_single_item(idx2)
            item['structure_1'] = item['structure']
            item['structure_2'] = item2['structure']
            return item
        return self._get_single_item(idx)

#--------------------------------------------------
# Data Preparation and Utility Functions
#--------------------------------------------------

def split_train_valid(df: Any, test_size: float = 0.3, seed: int = 42, label_column: Optional[str] = None) -> Tuple[Any, Any]:
    if label_column is None:
        train_df, valid_df = train_test_split(df, test_size=test_size, random_state=seed)
    else:
        train_df, valid_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[label_column])
    
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)

def split_train_valid_test(df: Any, seed: int = 42, label_column: Optional[str] = None) -> Tuple[Any, Any, Any]:
    if label_column is None:
        train_df, temp_df = split_train_valid(df, test_size=0.3, seed=seed)
        valid_df, test_df = split_train_valid(temp_df, test_size=0.5, seed=seed)
    else:
        train_df, temp_df = split_train_valid(df, test_size=0.3, seed=seed, label_column=label_column)
        valid_df, test_df = split_train_valid(temp_df, test_size=0.5, seed=seed, label_column=label_column)
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)
