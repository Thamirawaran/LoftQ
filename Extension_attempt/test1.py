import torch

def simulate_quantization(W, bits=4):
    """
    A simple simulated quantization function (fake quantization).
    In a real scenario, this would use NF4 or Int8 quantization.
    """
    # Normalize weight to [-1, 1]
    max_val = W.abs().max()
    W_norm = W / max_val
    
    # Quantize to 'bits' levels
    levels = 2 ** bits - 1
    W_quant = torch.round(W_norm * levels) / levels
    
    # De-normalize
    return W_quant * max_val

def get_adaptive_rank(W, energy_threshold=0.90, max_rank=64):
    """
    PROPOSED EXTENSION: Calculate the optimal rank based on Singular Value 'Energy'.
    """
    # 1. Perform SVD on the original High-Precision Weights
    # We only need singular values (S) here, so compute_uv=False is faster
    S = torch.linalg.svdvals(W)
    
    # 2. Calculate Total Energy (Sum of squares of singular values)
    total_energy = torch.sum(S ** 2)
    
    # 3. Find the minimum rank needed to capture 'energy_threshold' (e.g. 90%)
    current_energy = 0
    chosen_rank = 1
    
    for i in range(len(S)):
        current_energy += S[i] ** 2
        explained_variance = current_energy / total_energy
        
        if explained_variance >= energy_threshold:
            chosen_rank = i + 1
            break
            
    # Clip rank to be within reasonable bounds [1, max_rank]
    chosen_rank = min(chosen_rank, max_rank)
    return chosen_rank, explained_variance.item()

def run_loftq_algorithm(W, rank, num_steps=5):
    """
    The Standard LoftQ Algorithm (Algorithm 1 from the paper),
    but now using our ADAPTIVE rank.
    """
    # Initialize Adapters to Zero
    out_dim, in_dim = W.shape
    A = torch.zeros(in_dim, rank)
    B = torch.zeros(out_dim, rank)
    
    # Error tracking
    initial_err = torch.norm(W - simulate_quantization(W))
    
    # Alternating Optimization Loop
    for t in range(num_steps):
        # Step 1: Quantize the residual (W - current_adapter)
        # We simulate the quantized backbone Q
        # Note: B @ A.T gives the adapter contribution
        residual = W - (B @ A.T)
        Q = simulate_quantization(residual)
        
        # Step 2: SVD on the quantization error (W - Q)
        quantization_error = W - Q
        U, S, Vh = torch.linalg.svd(quantization_error, full_matrices=False)
        
        # Step 3: Update Adapters A and B using top-k singular values
        # A = U * sqrt(Sigma)
        # B = V * sqrt(Sigma)
        
        # Keep only top 'rank' components
        U_r = U[:, :rank]
        S_r = torch.diag(torch.sqrt(S[:rank]))
        Vh_r = Vh[:rank, :]
        
        # Update A and B
        # Note: Dimensions need to align with LoRA convention (A: d_in->r, B: r->d_out)
        # Here we simplify for demonstration: W approx Q + B @ A.T
        B = U_r @ S_r
        A = (S_r @ Vh_r).T 

    final_approx = Q + (B @ A.T)
    final_err = torch.norm(W - final_approx)
    
    return final_err, initial_err

# ==========================================
# PRESENTATION DEMO
# ==========================================

print("--- ADAPTIVE LOFTQ EXTENSION DEMO ---\n")

# 1. Create a 'Complex' Layer (High entropy, needs high rank)
# We simulate this by adding many random orthogonal components
torch.manual_seed(42)
W_complex = torch.randn(128, 128)

# 2. Create a 'Simple' Layer (Low entropy, needs low rank)
# We simulate this by creating a low-rank matrix explicitly
U_simple = torch.randn(128, 5)
V_simple = torch.randn(5, 128)
W_simple = U_simple @ V_simple + (torch.randn(128, 128) * 0.01) # Add slight noise

layers = {"Layer 1 (Complex)": W_complex, "Layer 2 (Simple)": W_simple}

for name, W in layers.items():
    print(f"Processing {name}...")
    
    # --- Step 1: Adaptive Rank Allocation ---
    # We ask for 90% information retention
    optimal_r, captured_var = get_adaptive_rank(W, energy_threshold=0.90)
    
    print(f"  > Standard Fixed Rank would be: r=16")
    print(f"  > Adaptive Method chose:        r={optimal_r}")
    print(f"  > Explained Variance:           {captured_var*100:.2f}%")
    
    # --- Step 2: Run LoftQ with Chosen Rank ---
    err_loftq, err_qonly = run_loftq_algorithm(W, rank=optimal_r)
    
    improvement = ((err_qonly - err_loftq) / err_qonly) * 100
    print(f"  > Reconstruction Error Reduced by: {improvement:.2f}%\n")