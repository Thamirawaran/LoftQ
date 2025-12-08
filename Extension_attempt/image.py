import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Re-create the Simulation Data ---
torch.manual_seed(42)
# Complex Layer (Slow decay)
W_complex = torch.randn(128, 128)
# Simple Layer (Fast decay)
U_simple = torch.randn(128, 5)
V_simple = torch.randn(5, 128)
W_simple = U_simple @ V_simple + (torch.randn(128, 128) * 0.01)

# --- 2. Calculate Cumulative Energy (Explained Variance) ---
def get_cumulative_energy(W):
    S = torch.linalg.svdvals(W)
    total_energy = torch.sum(S ** 2)
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
    return cumulative_energy.numpy() * 100  # Convert to %

y_complex = get_cumulative_energy(W_complex)
y_simple = get_cumulative_energy(W_simple)
x_axis = np.arange(1, 129)

# --- 3. Plotting the Chart ---
plt.figure(figsize=(10, 6), dpi=150)

# Plot Curves
plt.plot(x_axis, y_complex, label='Layer 1 (Complex)', color='#d62728', linewidth=2.5) # Red
plt.plot(x_axis, y_simple, label='Layer 2 (Simple)', color='#1f77b4', linewidth=2.5)   # Blue

# Plot The 90% Threshold Line
plt.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90% Information Threshold')

# Annotate the "Intersection" points (Where your adaptive rank was chosen)
# Complex Layer reaches 90% around rank 64
idx_complex = np.argmax(y_complex >= 90)
plt.scatter(idx_complex, y_complex[idx_complex], color='black', zorder=5)
plt.text(idx_complex+5, 85, f'Rank Needed: ~{idx_complex}\n(Standard r=16 Fails)', fontsize=10, color='#d62728')

# Simple Layer reaches 90% around rank 5
idx_simple = np.argmax(y_simple >= 90)
plt.scatter(idx_simple, y_simple[idx_simple], color='black', zorder=5)
plt.text(idx_simple+5, 92, f'Rank Needed: ~{idx_simple}\n(Standard r=16 Wasteful)', fontsize=10, color='#1f77b4')

# Styling
plt.title('Why Adaptive Rank is Needed: Singular Value Spectrum', fontsize=14, fontweight='bold')
plt.xlabel('Rank (Number of Parameters)', fontsize=12)
plt.ylabel('Cumulative Information Captured (%)', fontsize=12)
plt.xlim(0, 100) # Zoom in on the relevant part
plt.ylim(0, 105)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right')

# Save
plt.tight_layout()
plt.savefig('adaptive_loftq_chart.png')
plt.show()