import argparse
from datetime import datetime
import os
import pandas as pd

from utils import seed_everything, get_device, check_max_len, clean_gpu, prepare_self_datasets
from utils_config import get_model_config
from trainer import Trainer

def experiment(args_dict: dict) -> None:
    """
    Run the pretraining process for circRNA self-supervised learning.
    Proposed Method Tasks: MLM, SSP, and BPP (Pairing).
    """
    print('-' * 30)
    # Step 1: Seed and device setup
    seed_everything(args_dict['seed'])
    print(f"Seed set to: {args_dict['seed']}")
    device = get_device(args_dict['device'])
    print(f"Device: {device}")

    # Step 2: Define experiment name
    experiment_name = args_dict.get('exp', None)
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Experiment Name: {experiment_name}")
    print('-' * 30)

    # Step 3: Initialize Trainer
    print("[Step 1] Initializing Trainer for Pretraining")
    trainer = Trainer(
        seed=args_dict['seed'],
        device=device,
        experiment_name=experiment_name,
        verbose=args_dict['verbose']
    )

    # Step 4: Check max length for circRNA
    print("[Step 2] Loading Pretraining Data")
    df = pd.read_pickle('./data/' + args_dict['data_file'] + '.pkl')
    print(f"- Loaded dataset size: {len(df)}")
    df['length'] = df['circRNA'].apply(len)
    df = df[df['length'] <= args_dict['max_len']].reset_index(drop=True)
    print(f"- Filtered dataset size: {len(df)}")

    # Step 5: Split into train/valid/test and build circRNA + structure dataset
    print("[Step 3] Preparing Self-supervised Datasets")
    train_dataset, valid_dataset, test_dataset = prepare_self_datasets(
        df=df,
        max_len=args_dict['max_len'] + 2,
        seed=args_dict['seed'],
        kmer=args_dict['kmer'],
        is_test=False,  # Set to False for pretraining
    )

    if args_dict['verbose']:
        print(f"- Train size: {len(train_dataset)}")
        print(f"- Valid size: {len(valid_dataset)}")
        print(f"- Test size: {len(test_dataset)}")

    # Step 6: Set dataloaders
    trainer.set_dataloader_self(train_dataset, part=0, batch_size=args_dict['batch_size'], num_workers=args_dict['num_workers'])
    trainer.set_dataloader_self(valid_dataset, part=1, batch_size=args_dict['batch_size'], num_workers=args_dict['num_workers'])
    trainer.set_dataloader_self(test_dataset, part=2, batch_size=args_dict['batch_size'], num_workers=args_dict['num_workers'])

    # Step 7: Get model config (depends on model type)
    print("[Step 4] Configuring Model for Pretraining")
    config = get_model_config(
        model_name=args_dict['model_name'],
        d_model=args_dict['d_model'],
        n_layer=args_dict['n_layer'],
        verbose=args_dict['verbose'],
        rc=args_dict['rc'],
        vocab_size=train_dataset.vocab_size
    )

    # Step 8: Define model with pretraining mode = True
    trainer.define_model(
        config=config,
        model_name=args_dict['model_name'],
        pretrain=True,
        pooling_mode_target=args_dict['pooling_mode_target']
    )

    # Step 9: Set optimizer
    print("[Step 5] Setting Optimizer")
    trainer.set_optimizer(
        optimizer_name=args_dict['optimizer'],
        lr=args_dict['lr'],
        w_decay=args_dict.get('w_decay', None),
        freeze=args_dict['freeze']
    )
    clean_gpu()

    # Step 10: Pretraining execution with task flags
    print("[Step 6] Starting Pretraining (Proposed Tasks: MLM, SSP, BPP)")
    trainer.pretrain(
        epochs=args_dict['epochs'],
        earlystop=args_dict['earlystop'],
        mask_ratio=args_dict['mask_ratio'],
        mlm=args_dict['mlm'],
        ssp=args_dict['ssp'],
        pairing=args_dict['pairing'],
        rc=args_dict['rc'],
        log_name='pretrain',
        verbose=args_dict['verbose'],
        ssp_vocab_size=args_dict['ssp_vocab_size']
    )

    print("[Step 7] Pretraining Completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain circMAC model using MLM, SSP, and BPP (Pairing)")

    # Data settings
    parser.add_argument('--data_file', type=str, default='df_circ_ss', help='Path to circRNA data (pkl)')
    parser.add_argument('--max_len', type=int, default=1024, help='Max circRNA length')
    parser.add_argument('--kmer', type=int, default=1, help='k-mer size for circRNA')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Mask ratio for MLM')

    # Self-supervised tasks (Proposed)
    parser.add_argument('--mlm', action='store_true', help='Use MLM (Masked Language Modeling) task')
    parser.add_argument('--ssp', action='store_true', help='Use SSP (Secondary Structure Prediction) task')
    parser.add_argument('--pairing', action='store_true', help='Use BPP (Base Pairing Probability / Pairing) task')
    parser.add_argument('--ssp_vocab_size', type=int, default=4, help='SSP vocab size (e.g., 4 for 1-mer dot-bracket)')

    # Model settings
    parser.add_argument('--model_name', type=str, default='mamba',
                    choices=['mamba', 'transformer', 'hymba', 
                                'circmac', 'lstm',
                                'rnabert', 'rnaernie', 'rnafm', 'rnamsm'],
                    help='Model type')
    parser.add_argument('--rc', action='store_true', help='Use reverse complement')
    parser.add_argument('--d_model', type=int, default=64, help='Hidden dim')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers')
    parser.add_argument('--pooling_mode_target', type=str, default='mean', choices=['cls', 'mean'])

    # Optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num_workers')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--w_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=300, help='Max pretraining epochs')
    parser.add_argument('--earlystop', type=int, default=15, help='Early stop patience')
    parser.add_argument('--freeze', action='store_true', help='Freeze encoder layers')

    # Logging and control
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--device', type=int, default=-1, help='Device (-1 for CPU, 0~N for GPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exp', type=str, default=None, help='Experiment name')

    args = parser.parse_args()
    experiment(vars(args))
