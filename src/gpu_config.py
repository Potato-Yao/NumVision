"""
GPU Configuration and utilities for TensorFlow training.
"""
import tensorflow as tf


def check_gpu_availability():
    """
    Check if GPU is available and print GPU information.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    print("\n" + "="*60)
    print("GPU Configuration Check")
    print("="*60)
    
    # Check TensorFlow version
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n✓ GPU Available: {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            
            # Get GPU details if possible
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details:
                    print(f"    Details: {gpu_details}")
            except:
                pass
        
        # Check CUDA and cuDNN versions
        print(f"\nCUDA Available: {tf.test.is_built_with_cuda()}")
        print(f"GPU Device Name: {tf.test.gpu_device_name()}")
        
        return True
    else:
        print("\n✗ No GPU detected. Training will use CPU.")
        print("  To enable GPU training:")
        print("  1. Install CUDA Toolkit (11.8 or compatible)")
        print("  2. Install cuDNN (8.6 or compatible)")
        print("  3. Install tensorflow[and-cuda] or tensorflow-gpu")
        return False


def configure_gpu_memory_growth():
    """
    Configure GPU to use memory growth instead of allocating all memory at once.
    This prevents TensorFlow from allocating all GPU memory and allows sharing.
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\n✓ GPU memory growth enabled")
            print("  GPU will allocate memory as needed instead of all at once")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"\n✗ Error setting memory growth: {e}")


def set_mixed_precision(enable=True):
    """
    Enable mixed precision training (float16/float32) for better GPU performance.
    This can significantly speed up training on modern GPUs (V100, RTX 20xx+, A100, etc.)
    
    Args:
        enable: Whether to enable mixed precision
    """
    if enable:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("\n✓ Mixed Precision Training enabled (float16/float32)")
        print("  This can provide 2-3x speedup on modern GPUs")
        print(f"  Compute dtype: {policy.compute_dtype}")
        print(f"  Variable dtype: {policy.variable_dtype}")
    else:
        policy = tf.keras.mixed_precision.Policy('float32')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("\n✓ Using standard float32 precision")


def configure_for_training(use_mixed_precision=True, memory_growth=True):
    """
    Configure GPU settings optimally for training.
    
    Args:
        use_mixed_precision: Enable mixed precision for faster training
        memory_growth: Enable memory growth to prevent OOM errors
        
    Returns:
        bool: True if GPU is available and configured, False otherwise
    """
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    if gpu_available:
        # Configure memory growth
        if memory_growth:
            configure_gpu_memory_growth()
        
        # Configure mixed precision
        if use_mixed_precision:
            # Check if GPU supports mixed precision
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Mixed precision works best on GPUs with compute capability >= 7.0
                # (V100, T4, RTX 2000 series and newer)
                set_mixed_precision(enable=True)
        
        print("\n✓ GPU configuration complete!")
        print("="*60 + "\n")
    else:
        print("\n  CPU training will be used.")
        print("="*60 + "\n")
    
    return gpu_available


def limit_gpu_memory(memory_limit_mb):
    """
    Limit GPU memory usage to a specific amount in MB.
    Useful when sharing GPU with other processes.
    
    Args:
        memory_limit_mb: Memory limit in megabytes
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Set memory limit for the first GPU
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=memory_limit_mb)]
            )
            print(f"\n✓ GPU memory limited to {memory_limit_mb}MB")
        except RuntimeError as e:
            print(f"\n✗ Error setting memory limit: {e}")


def get_gpu_memory_info():
    """
    Get current GPU memory usage information.
    
    Returns:
        dict: Dictionary with memory information
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        return {"gpu_available": False}
    
    info = {
        "gpu_available": True,
        "num_gpus": len(gpus),
        "gpu_names": [gpu.name for gpu in gpus]
    }
    
    return info


def set_gpu_device(device_id=0):
    """
    Set specific GPU device to use (for multi-GPU systems).
    
    Args:
        device_id: GPU device ID to use (0, 1, 2, etc.)
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus and device_id < len(gpus):
        try:
            # Make only the specified GPU visible
            tf.config.set_visible_devices(gpus[device_id], 'GPU')
            print(f"\n✓ Using GPU {device_id}: {gpus[device_id].name}")
        except RuntimeError as e:
            print(f"\n✗ Error setting GPU device: {e}")
    else:
        print(f"\n✗ GPU {device_id} not available")


if __name__ == "__main__":
    # Test GPU configuration
    configure_for_training(use_mixed_precision=True, memory_growth=True)
    
    # Print additional GPU info
    info = get_gpu_memory_info()
    print("\nGPU Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

