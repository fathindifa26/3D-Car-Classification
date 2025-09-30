"""
Convert trained PyTorch model (.pth) to ONNX format for deployment
# Convert best checkpoint
python utils/convert_to_onnx.py --checkpoint experiments/vgg_lite_20250927_063311/best_checkpoint.pth

# Convert dengan custom output path
python utils/convert_to_onnx.py \
    --checkpoint experiments/vgg_lite_20250927_063311/best_checkpoint.pth \
    --output models/vgg_lite_best.onnx

# Convert dengan batch size tertentu (tidak dynamic)
python utils/convert_to_onnx.py \
    --checkpoint experiments/vgg_lite_20250927_063311/best_checkpoint.pth \
    --batch-size 8 \
    --no-dynamic
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import argparse
import json
import sys
import os

# Add parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from model.vgg import create_model
from config.config import Config

def convert_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 11,
    dynamic_axes: bool = True
):
    """
    Convert PyTorch model to ONNX format
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        output_path: Output path for .onnx file
        input_size: Input tensor size (batch, channels, height, width)
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic batch size
    """
    
    print(f"Converting model: {checkpoint_path}")
    print(f"Output path: {output_path}")
    
    # Load checkpoint
    device = torch.device('cpu')  # Use CPU for ONNX export
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        num_classes = config_dict.get('NUM_CLASSES', 5)
        dropout_rate = config_dict.get('DROPOUT_RATE', 0.3)
    else:
        # Default values
        num_classes = 5
        dropout_rate = 0.3
        print("Warning: No config found in checkpoint, using defaults")
    
    print(f"Model config - Classes: {num_classes}, Dropout: {dropout_rate}")
    
    # Create model
    model = create_model(
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        pytorch_output = model(dummy_input)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Set dynamic axes for variable batch size
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dynamic_axes_dict = None
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,                          # Model
        dummy_input,                    # Input tensor
        output_path,                    # Output path
        export_params=True,             # Store trained parameter weights
        opset_version=opset_version,    # ONNX version
        do_constant_folding=True,       # Optimize constant folding
        input_names=['input'],          # Input names
        output_names=['output'],        # Output names
        dynamic_axes=dynamic_axes_dict  # Dynamic axes
    )
    
    print(f"✓ ONNX model exported to: {output_path}")
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")
    
    # Test ONNX Runtime inference
    print("Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path)
    
    # Get input/output info
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    
    # Run inference
    ort_inputs = {input_name: dummy_input.numpy()}
    ort_output = ort_session.run([output_name], ort_inputs)[0]
    
    print(f"ONNX output shape: {ort_output.shape}")
    
    # Compare outputs
    pytorch_np = pytorch_output.detach().numpy()
    diff = np.abs(pytorch_np - ort_output).max()
    print(f"Max difference between PyTorch and ONNX: {diff}")
    
    if diff < 1e-5:
        print("✓ ONNX conversion successful - outputs match!")
    else:
        print("⚠ Warning: Large difference between PyTorch and ONNX outputs")
    
    # Save model info
    model_info = {
        'original_checkpoint': str(checkpoint_path),
        'onnx_path': str(output_path),
        'input_shape': [int(x) for x in input_size],
        'output_shape': [int(x) for x in ort_output.shape],
        'num_classes': int(num_classes),
        'dropout_rate': float(dropout_rate),
        'opset_version': int(opset_version),
        'dynamic_batch': bool(dynamic_axes),
        'max_difference': float(diff),
        'conversion_successful': bool(diff < 1e-5)
    }
    
    info_path = str(output_path).replace('.onnx', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✓ Model info saved to: {info_path}")
    
    return output_path, model_info

def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch checkpoint (.pth file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ONNX file path')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for export (default: 1)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size (default: 224)')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--no-dynamic', action='store_true',
                       help='Disable dynamic batch size')
    
    args = parser.parse_args()
    
    # Validate input
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    # Set output path
    if args.output is None:
        output_path = checkpoint_path.with_suffix('.onnx')
    else:
        output_path = Path(args.output)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set input size
    input_size = (args.batch_size, 3, args.image_size, args.image_size)
    
    print("="*60)
    print("PyTorch to ONNX Conversion")
    print("="*60)
    print(f"Input checkpoint: {checkpoint_path}")
    print(f"Output ONNX: {output_path}")
    print(f"Input size: {input_size}")
    print(f"Opset version: {args.opset}")
    print(f"Dynamic batch: {not args.no_dynamic}")
    print("="*60)
    
    try:
        convert_to_onnx(
            checkpoint_path=str(checkpoint_path),
            output_path=str(output_path),
            input_size=input_size,
            opset_version=args.opset,
            dynamic_axes=not args.no_dynamic
        )
        
        print("\n" + "="*60)
        print("Conversion completed successfully!")
        print(f"ONNX model: {output_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\nConversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()