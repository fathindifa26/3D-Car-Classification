import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
import shutil

class MultiLabelTrainer:
    """
    Trainer class for multi-label classification with VGG-lite
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Initialize tracking variables
        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        
        # Setup experiment directory and resume logic
        self.setup_experiment_directory()
        
        # Setup training components
        self.setup_training_components()
        
        # Load checkpoint if resuming
        if self.config.RESUME_FROM:
            self.load_checkpoint()
        
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        if self.config.RESUME_FROM:
            print(f"Resumed from epoch {self.start_epoch - 1}")
    
    def setup_experiment_directory(self):
        """Setup experiment directory with resume logic"""
        if self.config.RESUME_EXPERIMENT_DIR and Path(self.config.RESUME_EXPERIMENT_DIR).exists():
            # Resume existing experiment
            self.experiment_dir = Path(self.config.RESUME_EXPERIMENT_DIR)
            print(f"Resuming experiment: {self.experiment_dir}")
            
            # Initialize tensorboard writer (append mode)
            self.writer = SummaryWriter(
                log_dir=str(self.experiment_dir / 'tensorboard'), 
                purge_step=None  # Don't purge existing logs
            )
            
        else:
            # Create new experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = Path(self.config.SAVE_DIR) / f"vgg_lite_{timestamp}"
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            
            # Save config
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
            config_dict = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                          for k, v in config_dict.items()}
            
            with open(self.experiment_dir / 'config.json', 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Initialize tensorboard writer
            self.writer = SummaryWriter(log_dir=str(self.experiment_dir / 'tensorboard'))
    
    def setup_training_components(self):
        """Setup loss function, optimizer, and scheduler"""
        # Loss function
        if self.config.USE_CLASS_WEIGHTS:
            pos_weights = self.calculate_class_weights()
            print(f"Using class weights: {pos_weights}")
            self.pos_weights = pos_weights
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.BCELoss()
            self.pos_weights = None
        
        # Optimizer
        if self.config.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.OPTIMIZER}")
        
        # Scheduler
        if self.config.SCHEDULER == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.EPOCHS,
                eta_min=self.config.MIN_LR
            )
        elif self.config.SCHEDULER == 'cosine_restart':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.T_0,
                T_mult=getattr(self.config, 'T_MULT', 1),
                eta_min=self.config.MIN_LR
            )
        elif self.config.SCHEDULER == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.STEP_SIZE,
                gamma=self.config.GAMMA
            )
        else:
            self.scheduler = None
    
    def load_checkpoint(self):
        """Load checkpoint for resuming training"""
        checkpoint_path = Path(self.config.RESUME_FROM)
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint file not found: {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model state loaded")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Scheduler state loaded")
        
        # Load training progress
        self.start_epoch = checkpoint['epoch'] + 1
        
        if 'best_val_f1' in checkpoint:
            self.best_val_f1 = checkpoint['best_val_f1']
            print(f"✓ Best validation F1: {self.best_val_f1:.4f}")
        
        if 'metrics' in checkpoint:
            last_metrics = checkpoint['metrics']
            if 'loss' in last_metrics:
                self.best_val_loss = last_metrics['loss']
        
        # Load training history if available
        history_file = self.experiment_dir / 'training_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                self.train_losses = history.get('train_losses', [])
                self.val_losses = history.get('val_losses', [])
                self.train_f1_scores = history.get('train_f1_scores', [])
                self.val_f1_scores = history.get('val_f1_scores', [])
            print("✓ Training history loaded")
        
        print(f"✓ Checkpoint loaded. Resuming from epoch {self.start_epoch}")
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_f1_scores': self.train_f1_scores,
            'val_f1_scores': self.val_f1_scores,
            'epochs_completed': len(self.train_losses)
        }
        
        with open(self.experiment_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    def calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance"""
        all_labels = []
        print("Calculating class weights...")
        
        sample_count = 0
        max_samples = 1000
        
        for _, labels in self.train_loader:
            all_labels.append(labels)
            sample_count += len(labels)
            if sample_count >= max_samples:
                break
        
        all_labels = torch.cat(all_labels, dim=0)
        
        pos_counts = all_labels.sum(dim=0)
        total_samples = len(all_labels)
        
        pos_weights = []
        for i, pos_count in enumerate(pos_counts):
            neg_count = total_samples - pos_count
            if pos_count > 0:
                weight = neg_count / pos_count
            else:
                weight = 1.0
            pos_weights.append(weight.item())
        
        return torch.tensor(pos_weights, dtype=torch.float32).to(self.device)
    
    def compute_weighted_bce_loss(self, outputs, targets):
        """Compute weighted BCE loss manually"""
        if self.pos_weights is not None:
            loss = -(self.pos_weights * targets * torch.log(outputs + 1e-8) + 
                    (1 - targets) * torch.log(1 - outputs + 1e-8))
            return loss.mean()
        else:
            return self.criterion(outputs, targets)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            if self.pos_weights is not None:
                loss = self.compute_weighted_bce_loss(outputs, targets)
            else:
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            
            if hasattr(self.config, 'GRAD_CLIP') and self.config.GRAD_CLIP:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            if batch_idx % self.config.LOG_INTERVAL == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
        
        epoch_loss = running_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = epoch_loss
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                
                if self.pos_weights is not None:
                    loss = self.compute_weighted_bce_loss(outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                predictions = (outputs > 0.5).float()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        epoch_loss = running_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = epoch_loss
        
        return metrics
    
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate multi-label classification metrics"""
        pred_np = predictions.numpy()
        target_np = targets.numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_np, pred_np, average=None, zero_division=0
        )
        
        macro_f1 = precision_recall_fscore_support(
            target_np, pred_np, average='macro', zero_division=0
        )[2]
        
        micro_f1 = precision_recall_fscore_support(
            target_np, pred_np, average='micro', zero_division=0
        )[2]
        
        exact_match = accuracy_score(target_np, pred_np)
        hamming = hamming_loss(target_np, pred_np)
        
        metrics = {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'exact_match': exact_match,
            'hamming_loss': hamming,
            'precision_mean': precision.mean(),
            'recall_mean': recall.mean(),
            'f1_mean': f1.mean()
        }
        
        if hasattr(self.config, 'LABEL_NAMES'):
            for i, class_name in enumerate(self.config.LABEL_NAMES):
                metrics[f'{class_name}_precision'] = precision[i]
                metrics[f'{class_name}_recall'] = recall[i]
                metrics[f'{class_name}_f1'] = f1[i]
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        clean_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, 'item'):  # numpy scalar
                clean_metrics[k] = float(v.item())
            else:
                clean_metrics[k] = float(v) if isinstance(v, (int, float)) else v
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': clean_metrics,
            'best_val_f1': self.best_val_f1,
            'config': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                  for k, v in (self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)).items()}
            }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.experiment_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.experiment_dir / 'best_checkpoint.pth')
            print(f"New best model saved with val_f1: {metrics['macro_f1']:.4f}")
        
        # Save training history
        self.save_training_history()
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to tensorboard and console"""
        print(f"\nEpoch {epoch} Results:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['macro_f1']:.4f}")
        
        self.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        self.writer.add_scalars('F1_Score', {
            'train': train_metrics['macro_f1'],
            'val': val_metrics['macro_f1']
        }, epoch)
        
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        if hasattr(self.config, 'LABEL_NAMES'):
            for class_name in self.config.LABEL_NAMES:
                if f'{class_name}_f1' in train_metrics:
                    self.writer.add_scalars(f'F1_{class_name}', {
                        'train': train_metrics[f'{class_name}_f1'],
                        'val': val_metrics[f'{class_name}_f1']
                    }, epoch)
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        if self.start_epoch > 1:
            print(f"RESUMING training from epoch {self.start_epoch} to {self.config.EPOCHS}")
        else:
            print(f"STARTING training for {self.config.EPOCHS} epochs")
        print(f"{'='*60}")
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.config.EPOCHS + 1):
                epoch_start_time = time.time()
                
                # Train and validate
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate_epoch()
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Log metrics
                self.log_metrics(epoch, train_metrics, val_metrics)
                
                # Save metrics
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                self.train_f1_scores.append(train_metrics['macro_f1'])
                self.val_f1_scores.append(val_metrics['macro_f1'])
                
                # Check if best model
                is_best = val_metrics['macro_f1'] > self.best_val_f1
                if is_best:
                    self.best_val_f1 = val_metrics['macro_f1']
                    self.best_val_loss = val_metrics['loss']
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Print epoch time
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        
        except KeyboardInterrupt:
            print(f"\n{'='*60}")
            print("Training interrupted by user!")
            print(f"Progress saved. You can resume from epoch {epoch}")
            print(f"{'='*60}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
        self.writer.close()
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        if not self.train_losses:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # F1 curves
        ax2.plot(epochs, self.train_f1_scores, 'b-', label='Train F1')
        ax2.plot(epochs, self.val_f1_scores, 'r-', label='Val F1')
        ax2.set_title('Training and Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()