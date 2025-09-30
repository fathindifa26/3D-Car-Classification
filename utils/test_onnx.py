"""
Test ONNX model performance and accuracy on dataset
# Test basic accuracy dan FPS
python utils/test_onnx.py \
    --model experiments/vgg_lite_20250927_063311/best_checkpoint.onnx \
    --csv dataset/labels_3d_valid.csv \
    --images dataset/

# Test dengan benchmark speed
python utils/test_onnx.py \
    --model models/vgg_lite_best.onnx \
    --csv dataset/labels_3d_valid.csv \
    --images dataset/ \
    --benchmark \
    --benchmark-iterations 200

# Test dengan GPU (jika tersedia)
python utils/test_onnx.py \
    --model models/vgg_lite_best.onnx \
    --csv dataset/labels_3d_valid.csv \
    --images dataset/ \
    --providers CUDAExecutionProvider CPUExecutionProvider
"""

import onnxruntime as ort
import numpy as np
import torch
import time
import json
from pathlib import Path
import argparse
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
import sys

# Add parent directory to Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from data.dataloader import create_data_loaders

class ONNXTesterV2:
    """ONNX Tester using existing dataloader"""
    
    def __init__(self, onnx_path: str, providers: List[str] = None):
        """Initialize ONNX tester with dataloader"""
        self.onnx_path = onnx_path
        
        # Set providers
        if providers is None:
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        print(f"Loading ONNX model: {onnx_path}")
        print(f"Using providers: {providers}")
        
        # Create session
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        print(f"Input: {self.input_name} {self.input_shape}")
        print(f"Output: {self.output_name} {self.output_shape}")
        
        # Label names
        self.label_names = ['front_left', 'front_right', 'rear_left', 'rear_right', 'hood']
        
        # Timing statistics
        self.inference_times = []
    
    def predict_batch(self, images: torch.Tensor) -> tuple:
        """
        Predict batch of images
        
        Args:
            images: Batch of preprocessed images (torch.Tensor)
            
        Returns:
            (predictions, inference_time)
        """
        start_time = time.time()
        
        # Convert torch tensor to numpy and ensure float32
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy().astype(np.float32)
        else:
            images_np = images.astype(np.float32)
        
        # Run inference
        ort_inputs = {self.input_name: images_np}
        outputs = self.session.run([self.output_name], ort_inputs)
        predictions = outputs[0]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return predictions, inference_time
    
    def test_dataloader(self, dataloader, threshold: float = 0.5) -> Dict:
        """
        Test model using existing dataloader
        
        Args:
            dataloader: PyTorch DataLoader
            threshold: Classification threshold
            
        Returns:
            Test results dictionary
        """
        print(f"Testing with dataloader...")
        print(f"Dataset size: {len(dataloader.dataset)} samples")
        print(f"Batch size: {dataloader.batch_size}")
        
        # Initialize results
        all_predictions = []
        all_targets = []
        failed_batches = 0
        total_samples = 0
        
        # Test each batch
        for batch_idx, (images, targets) in enumerate(dataloader):
            try:
                # Predict batch
                predictions, inference_time = self.predict_batch(images)
                
                # Convert to binary predictions
                binary_preds = (predictions > threshold).astype(int)
                
                # Store results
                all_predictions.append(binary_preds)
                all_targets.append(targets.numpy())
                total_samples += len(images)
                
                # Progress
                if (batch_idx + 1) % 10 == 0:
                    samples_per_sec = len(images) / inference_time
                    print(f"Processed batch {batch_idx + 1}/{len(dataloader)}, "
                          f"Speed: {samples_per_sec:.1f} samples/sec")
                
            except Exception as e:
                print(f"Failed to process batch {batch_idx}: {e}")
                failed_batches += 1
        
        # Combine all results
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        print(f"Processing completed:")
        print(f"  Total samples: {total_samples}")
        print(f"  Failed batches: {failed_batches}")
        print(f"  Success rate: {(len(dataloader) - failed_batches) / len(dataloader) * 100:.1f}%")
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        # Calculate timing statistics
        batch_times = np.array(self.inference_times)
        samples_per_batch = dataloader.batch_size
        
        timing_stats = {
            'total_samples': int(total_samples),
            'failed_batches': int(failed_batches),
            'avg_batch_time': float(batch_times.mean()),
            'std_batch_time': float(batch_times.std()),
            'min_batch_time': float(batch_times.min()),
            'max_batch_time': float(batch_times.max()),
            'avg_samples_per_sec': float(samples_per_batch / batch_times.mean()),
            'max_samples_per_sec': float(samples_per_batch / batch_times.min()),
            'min_samples_per_sec': float(samples_per_batch / batch_times.max()),
            'avg_sample_time_ms': float(batch_times.mean() / samples_per_batch * 1000)
        }
        
        results = {
            'dataset_info': {
                'total_samples': len(dataloader.dataset),
                'processed_samples': total_samples,
                'batch_size': dataloader.batch_size,
                'num_batches': len(dataloader),
                'failed_batches': failed_batches,
                'threshold': threshold
            },
            'metrics': metrics,
            'timing': timing_stats,
            'model_info': {
                'onnx_path': self.onnx_path,
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'providers': self.session.get_providers()
            }
        }
        
        return results
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate classification metrics"""
        
        # Overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        macro_f1 = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )[2]
        
        micro_f1 = precision_recall_fscore_support(
            targets, predictions, average='micro', zero_division=0
        )[2]
        
        exact_match = accuracy_score(targets, predictions)
        hamming = hamming_loss(targets, predictions)
        
        metrics = {
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'exact_match_accuracy': float(exact_match),
            'hamming_loss': float(hamming),
            'precision_mean': float(precision.mean()),
            'recall_mean': float(recall.mean()),
            'f1_mean': float(f1.mean())
        }
        
        # Per-class metrics
        for i, label in enumerate(self.label_names):
            metrics[f'{label}_precision'] = float(precision[i])
            metrics[f'{label}_recall'] = float(recall[i])
            metrics[f'{label}_f1'] = float(f1[i])
        
        return metrics
    
    def benchmark_speed(self, batch_size: int = 32, num_iterations: int = 100) -> Dict:
        """Benchmark inference speed with dummy data"""
        
        print(f"Benchmarking speed: {num_iterations} iterations, batch_size={batch_size}")
        
        # Create dummy input matching training preprocessing
        input_shape = (batch_size, 3, 224, 224)  # Assuming 224x224 input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            self.predict_batch(dummy_input)
        
        # Clear previous timing stats
        self.inference_times = []
        
        # Benchmark
        print("Running benchmark...")
        for i in range(num_iterations):
            _, _ = self.predict_batch(dummy_input)
            
            if (i + 1) % 20 == 0:
                avg_time = np.mean(self.inference_times[-20:])
                samples_per_sec = batch_size / avg_time
                print(f"Iteration {i+1}/{num_iterations}, "
                      f"Time: {avg_time*1000:.1f}ms, "
                      f"Speed: {samples_per_sec:.1f} samples/sec")
        
        # Calculate statistics
        times = np.array(self.inference_times)
        
        benchmark_results = {
            'batch_size': batch_size,
            'num_iterations': num_iterations,
            'avg_time_ms': float(times.mean() * 1000),
            'std_time_ms': float(times.std() * 1000),
            'min_time_ms': float(times.min() * 1000),
            'max_time_ms': float(times.max() * 1000),
            'avg_samples_per_sec': float(batch_size / times.mean()),
            'max_samples_per_sec': float(batch_size / times.min()),
            'min_samples_per_sec': float(batch_size / times.max())
        }
        
        return benchmark_results


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test ONNX model with dataloader')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--images', type=str, required=True,
                       help='Path to images directory')
    parser.add_argument('--output', type=str, default='test_results/onnx_test_results_v2.json',
                       help='Output JSON file for results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for testing (default: 32)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run speed benchmark')
    parser.add_argument('--benchmark-iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--providers', nargs='+', 
                       choices=['CPUExecutionProvider', 'CUDAExecutionProvider'],
                       help='ONNX Runtime execution providers')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ONNX Model Testing (DataLoader Version)")
    print("="*60)
    
    # Create dataloader using existing function
    print("Creating test dataloader...")
    _, test_loader, _ = create_data_loaders(
        train_csv=args.csv,  # Use same CSV for consistency
        val_csv=args.csv,    # We only need test loader
        image_dir=args.images,
        batch_size=args.batch_size,
        image_size=224,
        num_workers=0,  # Disable multiprocessing for testing
        augment=False   # No augmentation for testing
    )
    
    # Use validation loader for testing (no augmentation)
    print(f"Test dataset: {len(test_loader.dataset)} samples")
    print(f"Batch size: {test_loader.batch_size}")
    
    # Initialize tester
    tester = ONNXTesterV2(args.model, args.providers)
    
    results = {}
    
    # Test on dataset
    print("\n1. Testing on dataset...")
    dataset_results = tester.test_dataloader(test_loader, args.threshold)
    results['dataset_test'] = dataset_results
    
    # Print results
    metrics = dataset_results['metrics']
    timing = dataset_results['timing']
    
    print(f"\nDataset Test Results:")
    print(f"  Processed: {dataset_results['dataset_info']['processed_samples']}/{dataset_results['dataset_info']['total_samples']} samples")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Exact Match: {metrics['exact_match_accuracy']:.4f}")
    print(f"  Speed: {timing['avg_samples_per_sec']:.1f} samples/sec")
    print(f"  Per-sample time: {timing['avg_sample_time_ms']:.1f}ms")
    
    # Speed benchmark
    if args.benchmark:
        print("\n2. Running speed benchmark...")
        benchmark_results = tester.benchmark_speed(args.batch_size, args.benchmark_iterations)
        results['speed_benchmark'] = benchmark_results
        
        print(f"\nSpeed Benchmark Results:")
        print(f"  Avg Time: {benchmark_results['avg_time_ms']:.1f}ms/batch")
        print(f"  Avg Speed: {benchmark_results['avg_samples_per_sec']:.1f} samples/sec")
        print(f"  Max Speed: {benchmark_results['max_samples_per_sec']:.1f} samples/sec")
    
    # Save results
    print(f"\n3. Saving results to: {args.output}")
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Testing completed!")
    print(f"Results saved to: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()