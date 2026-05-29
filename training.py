import argparse
from datetime import datetime
import os
import pandas as pd

from utils import seed_everything, get_device, check_max_len, clean_gpu
from utils_config import get_model_config
from trainer import Trainer
from utils import prepare_datasets


def experiment(args_dict: dict) -> None:
    """
    Run training based on provided arguments.
    """
    # Step 0. Set up environment
    print('-' * 30)
    print('[Step 0] Setting up environment')
    seed_everything(args_dict['seed'])
    print(f"Seed set to: {args_dict['seed']}")

    device = get_device(args_dict['device'])
    print(f"Device: {device}")

    target = args_dict['target'].lower()
    print(f"Target task: {target}")

    experiment_name = args_dict.get('exp', None)
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Experiment Name: {experiment_name}")
    print('-' * 30)
    
    # Step 1. Initialize Trainer
    print('[Step 1] Initializing Trainer')
    trainer = Trainer(
        seed=args_dict['seed'],
        device=device,
        experiment_name=experiment_name,
        verbose=args_dict['verbose'],
    )
    
    # Step 2. Load Data
    print('[Step 2] Loading Data for training')
    if target == 'mirna':
        max_len = check_max_len(args_dict['max_len'], args_dict['model_name'])
        df = pd.read_pickle(f'./data/df_train_final.pkl')
        df_test = pd.read_pickle(f'./data/df_test_final.pkl')
    else:
        raise ValueError(f"Unrecognized target: '{target}' (Proposed: mirna only)")
    
    df['length'] = df['circRNA'].apply(len)
    df_test['length'] = df_test['circRNA'].apply(len)

    df = df[df['length'] <= max_len]
    df_test = df_test[df_test['length'] <= max_len]

    # Task is always 'sites'
    df = df[df['binding'] == 1].reset_index(drop=True)
    df_test = df_test[df_test['binding'] == 1].reset_index(drop=True)

    if args_dict['verbose']:
        print(f"Data loaded: {len(df)} records")
        print(f"- df_test: {len(df_test)}")

    # Step 3. Prepare Datasets
    print('[Step 3] Setting Datasets for training')
    train_dataset, valid_dataset, test_dataset, extra_dataset = prepare_datasets(
            df=df, 
            df_test=df_test,
            max_len=max_len + 2,  # 2 for special tokens (CLS and EOS)
            target=target, 
            seed=args_dict['seed'],
            kmer=args_dict['kmer'],
        )

    if args_dict['verbose']:
        total_size = len(train_dataset) + len(valid_dataset) + len(test_dataset)
        print(f"Total dataset size: {total_size}")
        sample = train_dataset.__getitem__(0)
        print(sample['circRNA'].shape, sample['target'].shape)
        print(f"- Train dataset: {len(train_dataset)}, ratio: {len(train_dataset)/total_size:.4f}")
        print(f"- Valid dataset: {len(valid_dataset)}, ratio: {len(valid_dataset)/total_size:.4f}")
        print(f"- Test dataset: {len(test_dataset)}, ratio: {len(test_dataset)/total_size:.4f}")

    trainer.set_dataloader(train_dataset, part=0, batch_size=args_dict['batch_size'], num_workers=args_dict['num_workers'])
    trainer.set_dataloader(valid_dataset, part=1, batch_size=args_dict['batch_size'], num_workers=args_dict['num_workers'])
    trainer.set_dataloader(test_dataset, part=2, batch_size=args_dict['batch_size'], num_workers=args_dict['num_workers'])

    if extra_dataset is not None:
        trainer.set_dataloader(extra_dataset, part=3, batch_size=args_dict['batch_size'], num_workers=args_dict['num_workers'])
        if args_dict['verbose']:
            print(f"- Extra dataset: {len(extra_dataset)}, ratio: {len(extra_dataset)/total_size:.4f}")
    
    # Step 4. Configure Model
    print('[Step 4] Configuring Model for training')
    config = get_model_config(
        model_name=args_dict['model_name'],
        d_model=args_dict['d_model'],
        n_layer=args_dict['n_layer'],
        verbose=args_dict['verbose'],
        rc=args_dict['rc'],
        vocab_size=train_dataset.vocab_size
    )
    if args_dict['model_name'] in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']:
        config.trainable = True if args_dict['trainable_pretrained'] else False
        if args_dict['verbose']:
            print(f"- Pretrained model trainable: {config.trainable}")

    # Apply CircMAC Ablation flags
    if args_dict['model_name'] == 'circmac':
        if hasattr(config, 'use_attn'):
            config.use_attn = not args_dict.get('no_attn', False)
        if hasattr(config, 'use_mamba'):
            config.use_mamba = not args_dict.get('no_mamba', False)
        if hasattr(config, 'use_conv'):
            config.use_conv = not args_dict.get('no_conv', False)
        # Set ablation flags directly on config for model to read
        config.no_circular_rel_bias = args_dict.get('no_circular_rel_bias', False)
        config.no_attn = args_dict.get('no_attn', False)
        config.no_mamba = args_dict.get('no_mamba', False)
        config.no_conv = args_dict.get('no_conv', False)

    is_pretrained = True if args_dict['load_pretrained'] is not None else False
        
    trainer.define_model(
        config=config,
        model_name=args_dict['model_name'],
        pretrain=is_pretrained,
        pooling_mode_target=args_dict['pooling_mode_target'],
        is_convblock=args_dict['is_convblock'],
        is_cross_attention=args_dict['interaction'] == 'cross_attention',
        interaction=args_dict['interaction'],
        use_unified_head=args_dict.get('use_unified_head', False),
        binding_pooling=args_dict.get('binding_pooling', 'mean'),
        site_head_type=args_dict.get('site_head_type', 'conv1d'),
    )

    # Step 4-1. Set Pretrained circRNA Encoder (only when frozen, i.e. not trainable)
    if args_dict['model_name'] in ['rnabert', 'rnaernie', 'rnafm', 'rnamsm']:
        if not args_dict.get('trainable_pretrained', False):
            print(f'[Step 4-1] Loading pretrained circRNA encoder (frozen): {args_dict["model_name"]}')
            trainer.define_pretrained_model(model_name=args_dict['model_name'])
        else:
            print(f'[Step 4-1] Pretrained model is trainable, using internal encoder')

    # Step 5. Set Pretrained Target Model (for target sequences such as miRNA or protein)
    print('[Step 5] Setting Pretrained Target (miRNA or RBP)')
    trainer.set_pretrained_target(target=args_dict['target'], rna_model=args_dict['target_model'])

    # (Optionally, load trained or pretrained weights)
    if args_dict['load_trained']:
        print('[Step 5-1] Loading trained model weights')
        trainer.load_model_from_path(args_dict['load_trained'], verbose=args_dict['verbose'])
    elif args_dict['load_pretrained']:
        print(f'[Step 5-2] Loading pretrained model weights from: {args_dict["load_pretrained"]}')
        trainer.load_model_from_path(args_dict['load_pretrained'], verbose=args_dict['verbose'])

    # Step 6. Set Optimizer
    print('[Step 6] Setting Optimizer')
    trainer.set_optimizer(
        optimizer_name=args_dict['optimizer'],
        lr=args_dict['lr'],
        w_decay=args_dict.get('w_decay', None),
        freeze=args_dict['freeze']
    )
    clean_gpu()

    # Step 7. Start Training
    print('[Step 7] Starting Training')
    logs = trainer.train(
        epochs=args_dict['epochs'],
        earlystop=args_dict['earlystop'],
        task=args_dict['task'],
        forward_mode=args_dict['forward_mode'],
        rc=args_dict['rc'],
        log_name='training'
    )
    trainer.save_model(pretrain=False)
    print('[Step 8] Training Completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a circRNA model using ModelWrapper")
    
    # Target settings
    parser.add_argument('--target', type=str, default='mirna', choices=['mirna'], help='Target type (Proposed: mirna only)')
    parser.add_argument('--target_model', type=str, default='rnabert', choices=['rnabert', 'rnaernie', 'rnafm', 'rnamsm'], help='Target model')
    parser.add_argument('--pooling_mode_target', type=str, default='mean', choices=['cls', 'mean', 'None'], help='Pooling mode for target')

    # Data settings
    parser.add_argument('--task', type=str, default='sites', choices=['sites'], help='Training task (Proposed: sites only)')
    parser.add_argument('--forward_mode', type=int, default=1, choices=[1, 2, 3], help='Forward mode for training')
    parser.add_argument('--AGO', type=int, default=1, choices=[1, 3, 5], help='AGO threshold')
    parser.add_argument('--max_len', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--kmer', type=int, default=1, help='K-mer size')

    # Model settings
    parser.add_argument('--model_name', type=str, default='mamba',
                        choices=['mamba', 'transformer', 'hymba', 
                                 'circmac', 'lstm',
                                 'rnabert', 'rnaernie', 'rnafm', 'rnamsm'],
                        help='Model type')

    parser.add_argument('--rc', action='store_true', help='Use reverse complement')
    parser.add_argument('--d_model', type=int, default=64, help='Model hidden dimension')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers')
    parser.add_argument('--trainable_pretrained', action='store_true', help='Allow training of pretrained model')
    parser.add_argument('--is_convblock', action='store_true', help='Use ConvBlock in model')
    parser.add_argument('--is_cross_attention', action='store_true', help='Use cross attention in model (legacy, use --interaction instead)')
    parser.add_argument('--interaction', type=str, default='concat', choices=['concat', 'elementwise', 'cross_attention'], help='Interaction mechanism between circRNA and miRNA')
    parser.add_argument('--use_unified_head', action='store_true', help='Use unified site head (site-first approach)')
    parser.add_argument('--binding_pooling', type=str, default='mean', choices=['mean', 'max', 'attention'], help='Pooling method for deriving binding from sites')
    parser.add_argument('--site_head_type', type=str, default='conv1d', choices=['conv1d', 'linear'], help='Site head classifier type')

    # CircMAC Ablation settings
    parser.add_argument('--no_circular_rel_bias', action='store_true', help='Disable circular relative bias in attention')
    parser.add_argument('--no_attn', action='store_true', help='Disable Attention branch in CircMAC')
    parser.add_argument('--no_mamba', action='store_true', help='Disable Mamba branch in CircMAC')
    parser.add_argument('--no_conv', action='store_true', help='Disable Conv branch in CircMAC')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num_workers')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--earlystop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--device', type=int, default=-1, help='Device ID (-1 for CPU, 0~N for GPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Experiment and model loading settings
    parser.add_argument('--exp', type=str, default=None, help='Experiment name (optional)')
    parser.add_argument('--load_pretrained', type=str, default=None, help='Load pretrained model experiment name')
    parser.add_argument('--load_trained', type=str, default=None, help='Load trained model experiment name')
    parser.add_argument('--freeze', action='store_true', help='Freeze pretrained model weights')

    args = parser.parse_args()
    args_dict = vars(args)

    # Backward compat: --is_cross_attention overrides --interaction
    if args_dict['is_cross_attention']:
        args_dict['interaction'] = 'cross_attention'

    experiment(args_dict)
