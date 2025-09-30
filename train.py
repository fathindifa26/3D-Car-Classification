"""
Clean training script for VGG-lite multi-label classification with resume capability
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from config.config import Config, FastConfig, ProductionConfig
from data.dataloader import create_data_loaders
from model.vgg import create_model
from trainer.trainer import MultiLabelTrainer


def find_latest_experiment():
    """Find the latest experiment directory"""
    exp_dir = Path('experiments')
    if not exp_dir.exists():
        return None
    
    experiments = list(exp_dir.glob('vgg_lite_*'))
    if not experiments:
        return None
    
    # Sort by modification time, get the latest
    latest = max(experiments, key=lambda x: x.stat().st_mtime)
    return str(latest)


def main():
    """Main training function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train VGG-lite multi-label classifier')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint file or experiment directory to resume from')
    parser.add_argument('--resume-latest', action='store_true',
                       help='Resume from the latest experiment')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'fast', 'production'],
                       help='Configuration to use')
    
    args = parser.parse_args()
    
    # Choose configuration
    if args.config == 'fast':
        config = FastConfig
    elif args.config == 'production':
        config = ProductionConfig
    else:
        config = Config
    
    # Handle resume logic
    if args.resume_latest:
        latest_exp = find_latest_experiment()
        if latest_exp:
            config.set_resume(experiment_dir=latest_exp)
            print(f"Will resume from latest experiment: {latest_exp}")
        else:
            print("No previous experiments found. Starting new training.")
    
    elif args.resume:
        resume_path = Path(args.resume)
        
        if resume_path.is_dir():
            # Resume from experiment directory
            config.set_resume(experiment_dir=str(resume_path))
            print(f"Will resume from experiment directory: {resume_path}")
        
        elif resume_path.is_file() and resume_path.suffix == '.pth':
            # Resume from specific checkpoint file
            config.set_resume(checkpoint_path=str(resume_path))
            print(f"Will resume from checkpoint: {resume_path}")
        
        else:
            print(f"Error: Resume path not found or invalid: {resume_path}")
            return
    
    # Update paths if needed
    config.update_paths(
        image_dir='dataset/',
        train_csv='dataset/labels_3d.csv',
        val_csv='dataset/labels_3d_valid.csv'
    )
    
    # Validate paths exist
    try:
        config.validate_paths()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    print("="*60)
    print("VGG-lite Multi-Label Car Door Classification Training")
    if config.RESUME_FROM:
        print(f"RESUMING from: {config.RESUME_FROM}")
    print("="*60)
    
    # Create data loaders
    print("\n1. Creating data loaders...")
    train_loader, val_loader, train_dataset = create_data_loaders(
        train_csv=config.TRAIN_CSV,
        val_csv=config.VAL_CSV,
        image_dir=config.IMAGE_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS,
        augment=config.AUGMENT_DATA
    )
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    
    # Print label distribution
    print(f"\nLabel Distribution:")
    distribution = train_dataset.get_label_distribution()
    for label, stats in distribution.items():
        print(f"  {label}: {stats['positive']:>4}/{len(train_dataset):>4} ({stats['ratio']:>6.2%})")
    
    # Create model
    print(f"\n2. Creating model...")
    model = create_model(
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)
    
    print(f"  Model: VGG-lite")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Dropout rate: {config.DROPOUT_RATE}")
    
    # Create trainer
    print(f"\n3. Setting up trainer...")
    trainer = MultiLabelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.DEVICE
    )
    
    print(f"  Optimizer: {config.OPTIMIZER}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Scheduler: {config.SCHEDULER}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Class weights: {config.USE_CLASS_WEIGHTS}")
    
    # Start training
    print(f"\n4. Starting training...")
    print("="*60)
    
    try:
        trainer.train()
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Results saved to: {trainer.experiment_dir}")
        print(f"Best validation F1: {trainer.best_val_f1:.4f}")
        print(f"TensorBoard logs: {trainer.experiment_dir}/tensorboard")
        print("="*60)
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted. You can resume with:")
        print(f"python train.py --resume {trainer.experiment_dir}")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()