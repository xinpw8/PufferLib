"""
GPU-optimized sparse-to-dense conversion for chess action masks.

This module provides vectorized PyTorch operations to convert sparse action masks
to dense format without Python loops, maintaining performance while reducing
memory bandwidth by 55%.
"""
import torch


def sparse_to_dense_gpu(observations, num_actions=1968, max_legal_moves=64):
    """
    Convert sparse action masks to dense format using vectorized PyTorch operations.
    
    Args:
        observations: Tensor of shape [batch_size, obs_size] where obs_size = 1537
                     Format: [board_state(1472)] + [num_legal_moves(1)] + [legal_action_ids(64)]
        num_actions: Total number of possible actions (1968 for chess)
        max_legal_moves: Maximum number of legal moves in sparse representation (64)
    
    Returns:
        dense_masks: Tensor of shape [batch_size, num_actions] with 1.0 for legal moves
    """
    device = observations.device
    batch_size = observations.shape[0]
    
    # Extract sparse mask components
    sparse_start = 1472
    num_legal_moves = observations[:, sparse_start].long()  # [batch_size]
    action_ids = observations[:, sparse_start+1:sparse_start+1+max_legal_moves].long()  # [batch_size, 64]
    
    # Create dense mask initialized to zeros
    dense_masks = torch.zeros(batch_size, num_actions, device=device, dtype=torch.float32)
    
    # Create batch indices for advanced indexing
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # [batch_size, 1]
    
    # Create mask for valid action IDs (where count < num_legal_moves for each batch item)
    move_indices = torch.arange(max_legal_moves, device=device).unsqueeze(0)  # [1, 64]
    valid_mask = move_indices < num_legal_moves.unsqueeze(1)  # [batch_size, 64]
    
    # Extract valid action IDs and their corresponding batch indices
    valid_actions = action_ids[valid_mask]  # [total_valid_moves]
    valid_batch_indices = batch_indices.expand(-1, max_legal_moves)[valid_mask]  # [total_valid_moves]
    
    # Clamp action IDs to valid range to prevent out-of-bounds access
    valid_actions = torch.clamp(valid_actions, 0, num_actions - 1)
    
    # Use advanced indexing to set legal moves to 1.0
    dense_masks[valid_batch_indices, valid_actions] = 1.0
    
    return dense_masks


def sparse_to_dense_simple(observations, num_actions=1968, max_legal_moves=64):
    """
    Simpler implementation using scatter for comparison.
    
    This version may be slower but is easier to understand and debug.
    """
    device = observations.device
    batch_size = observations.shape[0]
    
    # Extract sparse mask components  
    sparse_start = 1472
    num_legal_moves = observations[:, sparse_start].long()
    action_ids = observations[:, sparse_start+1:sparse_start+1+max_legal_moves].long()
    
    # Create dense mask
    dense_masks = torch.zeros(batch_size, num_actions, device=device, dtype=torch.float32)
    
    # Use scatter to set legal moves
    for i in range(batch_size):
        n_moves = num_legal_moves[i]
        if n_moves > 0:
            valid_actions = action_ids[i, :n_moves]
            # Clamp to valid range
            valid_actions = torch.clamp(valid_actions, 0, num_actions - 1)
            dense_masks[i, valid_actions] = 1.0
    
    return dense_masks


# Benchmark function to compare performance
def benchmark_conversion(batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Benchmark sparse-to-dense conversion performance.
    """
    import time
    
    # Create sample sparse observations
    observations = torch.randn(batch_size, 1537, device=device)
    # Set up realistic sparse masks
    observations[:, 1472] = torch.randint(10, 30, (batch_size,), device=device).float()  # 10-30 legal moves
    for i in range(batch_size):
        n_moves = int(observations[i, 1472])
        observations[i, 1473:1473+n_moves] = torch.randint(0, 1968, (n_moves,), device=device).float()
    
    # Warmup
    for _ in range(10):
        _ = sparse_to_dense_gpu(observations)
    
    # Benchmark GPU version
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(100):
        result_gpu = sparse_to_dense_gpu(observations)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    gpu_time = time.time() - start_time
    
    # Benchmark simple version for comparison
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    for _ in range(100):
        result_simple = sparse_to_dense_simple(observations)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    simple_time = time.time() - start_time
    
    print(f"Batch size: {batch_size}, Device: {device}")
    print(f"GPU optimized: {gpu_time:.4f}s")
    print(f"Simple version: {simple_time:.4f}s") 
    print(f"Speedup: {simple_time/gpu_time:.2f}x")
    
    # Verify results are identical
    if torch.allclose(result_gpu, result_simple):
        print("✅ Results match")
    else:
        print("❌ Results differ!")
    
    return gpu_time, simple_time


if __name__ == "__main__":
    # Run benchmarks
    print("Benchmarking sparse-to-dense conversion...")
    benchmark_conversion(batch_size=32)
    benchmark_conversion(batch_size=128)
    benchmark_conversion(batch_size=512)