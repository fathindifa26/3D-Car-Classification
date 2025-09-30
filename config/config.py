"""
Configuration file for VGG-lite multi-label classification training
"""

import torch
from pathlib import Path

class Config:
    """Training configuration"""
    
    # Data paths
    IMAGE_DIR = 'dataset/'
    TRAIN_CSV = 'dataset/labels_3d.csv'  # You specify this
    VAL_CSV = 'dataset/labels_3d_valid.csv'      # You specify this
    
    # Data parameters
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    NUM_CLASSES = 5

    # Model parameters
    DROPOUT_RATE = 0.2 # awalnya 0.3
    
    # Training parameters
    EPOCHS = 1000
    LEARNING_RATE = 1e-3
    OPTIMIZER = 'adamw'  # 'adam' or 'adamw'
    WEIGHT_DECAY = 1e-4 # awalnya 1e-4
    
    # Learning rate scheduler
    SCHEDULER = 'cosine_restart'  # 'cosine', 'cosine_restart', 'step', or 'none'
    MIN_LR = 1e-6                # Minimum learning rate for all schedulers
    
    # Cosine restart specific parameters
    T_0 = 50                     # Restart every 50 epochs
    T_MULT = 1                   # Keep same cycle length
    
    # Step scheduler specific parameters  
    STEP_SIZE = 15               # Step size for StepLR
    GAMMA = 0.1                  # Gamma for StepLR

    # Training options
    GRAD_CLIP = 1.0
    USE_CLASS_WEIGHTS = False
    AUGMENT_DATA = True
    
    # Logging and saving
    LOG_INTERVAL = 50
    SAVE_DIR = 'experiments'
    
    # Resume training
    RESUME_FROM = None  # Path to checkpoint file or 'auto' for latest
    RESUME_EXPERIMENT_DIR = None  # Path to experiment directory to resume
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Label names (for logging)
    LABEL_NAMES = ['front_left', 'front_right', 'rear_left', 'rear_right', 'hood']
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary for saving"""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
    
    @classmethod
    def update_paths(cls, image_dir=None, train_csv=None, val_csv=None):
        """Update data paths"""
        if image_dir:
            cls.IMAGE_DIR = image_dir
        if train_csv:
            cls.TRAIN_CSV = train_csv
        if val_csv:
            cls.VAL_CSV = val_csv
    
    @classmethod
    def set_resume(cls, checkpoint_path=None, experiment_dir=None):
        """Set resume parameters"""
        if experiment_dir:
            cls.RESUME_EXPERIMENT_DIR = experiment_dir
            # Auto-detect checkpoint file
            exp_path = Path(experiment_dir)
            if (exp_path / 'best_checkpoint.pth').exists():
                cls.RESUME_FROM = str(exp_path / 'best_checkpoint.pth')
            elif (exp_path / 'latest_checkpoint.pth').exists():
                cls.RESUME_FROM = str(exp_path / 'latest_checkpoint.pth')
        elif checkpoint_path:
            cls.RESUME_FROM = checkpoint_path
            cls.RESUME_EXPERIMENT_DIR = str(Path(checkpoint_path).parent)
    
    @classmethod
    def validate_paths(cls):
        """Validate that all paths exist"""
        paths_to_check = [
            (cls.IMAGE_DIR, "Image directory"),
            (cls.TRAIN_CSV, "Training CSV"),
            (cls.VAL_CSV, "Validation CSV")
        ]
        
        for path, description in paths_to_check:
            if not Path(path).exists():
                raise FileNotFoundError(f"{description} not found: {path}")
        
        print(f"✓ All paths validated:")
        print(f"  Image dir: {cls.IMAGE_DIR}")
        print(f"  Train CSV: {cls.TRAIN_CSV}")
        print(f"  Val CSV: {cls.VAL_CSV}")


# Alternative configurations for different experiments
class FastConfig(Config):
    """Fast training config for testing"""
    EPOCHS = 5
    BATCH_SIZE = 16
    LOG_INTERVAL = 10


class ProductionConfig(Config):
    """Production config with more robust settings"""
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-3
    SCHEDULER = 'cosine'
    USE_CLASS_WEIGHTS = True