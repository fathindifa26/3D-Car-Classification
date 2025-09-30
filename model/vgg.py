import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class VGGLiteBlock(nn.Module):
    """
    VGG-like convolutional block with Conv → ReLU → BatchNorm → MaxPool
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2
    ):
        super(VGGLiteBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False  # No bias when using BatchNorm
        )
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.max_pool(x)
        return x


class VGGLiteClassifier(nn.Module):
    """
    VGG-lite CNN for multi-label classification
    
    Architecture:
    Conv(32, 3x3) → ReLU → BatchNorm → MaxPool(2x2)
    Conv(64, 3x3) → ReLU → BatchNorm → MaxPool(2x2)
    Conv(128, 3x3) → ReLU → BatchNorm → MaxPool(2x2)
    Conv(256, 3x3) → ReLU → BatchNorm → GlobalAveragePool
    Dense(128) → ReLU → Dropout(0.3)
    Dense(64) → ReLU → Dropout(0.3)
    Dense(5, activation=sigmoid)
    """
    
    def __init__(
        self, 
        num_classes: int = 5, 
        input_channels: int = 3,
        dropout_rate: float = 0.3
    ):
        super(VGGLiteClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv_block1 = VGGLiteBlock(input_channels, 32)  # 224x224 → 112x112
        self.conv_block2 = VGGLiteBlock(32, 64)              # 112x112 → 56x56
        self.conv_block3 = VGGLiteBlock(64, 128)             # 56x56 → 28x28
        
        # Final conv layer without pooling
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, num_classes) with sigmoid activation
        """
        # Convolutional blocks
        x = self.conv_block1(x)  # (B, 32, 112, 112)
        x = self.conv_block2(x)  # (B, 64, 56, 56)
        x = self.conv_block3(x)  # (B, 128, 28, 28)
        
        # Final conv layer
        x = self.conv4(x)        # (B, 256, 28, 28)
        x = self.relu4(x)
        x = self.batch_norm4(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 256)
        
        # Fully connected layers
        x = self.fc1(x)          # (B, 128)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)          # (B, 64)
        x = self.relu_fc2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)          # (B, num_classes)
        x = self.sigmoid(x)      # Multi-label classification
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Get intermediate feature maps for visualization
        
        Returns:
            Dictionary with feature maps from each block
        """
        features = {}
        
        x = self.conv_block1(x)
        features['block1'] = x
        
        x = self.conv_block2(x)
        features['block2'] = x
        
        x = self.conv_block3(x)
        features['block3'] = x
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.batch_norm4(x)
        features['block4'] = x
        
        return features
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    num_classes: int = 5,
    input_channels: int = 3,
    dropout_rate: float = 0.3,
    pretrained: bool = False
) -> VGGLiteClassifier:
    """
    Create VGG-lite model
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (3 for RGB)
        dropout_rate: Dropout rate for regularization
        pretrained: Whether to load pretrained weights (not implemented yet)
    
    Returns:
        VGGLiteClassifier model
    """
    model = VGGLiteClassifier(
        num_classes=num_classes,
        input_channels=input_channels,
        dropout_rate=dropout_rate
    )
    
    if pretrained:
        # TODO: Implement pretrained weights loading
        print("Pretrained weights not implemented yet")
    
    return model


# Model summary function
def model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].type())
            summary[m_key]["output_shape"] = list(output.shape)
            
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    device = next(model.parameters()).device
    summary = {}
    hooks = []

    model.apply(register_hook)
    
    # Create input tensor
    x = torch.randn(1, *input_size).to(device)
    model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    print("="*70)
    print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
    print("="*70)
    
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        # Input shape
        line_new = f"{layer:<25} {str(summary[layer]['output_shape']):<25} {summary[layer]['nb_params']:,}"
        total_params += summary[layer]["nb_params"]
        
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    print("="*70)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {(total_params - trainable_params):,}")
    print("="*70)