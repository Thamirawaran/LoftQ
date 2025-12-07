"""
Plot training loss from train.txt log file
"""
import matplotlib.pyplot as plt
import re
import json

# Read the log file
log_file = 'train.txt'

epochs = []
losses = []

with open(log_file, 'r') as f:
    for line in f:
        # Look for lines containing loss information (starts with {'loss':)
        if line.strip().startswith("{'loss':"):
            try:
                # Replace single quotes with double quotes for JSON parsing
                line_json = line.strip().replace("'", '"')
                data = json.loads(line_json)
                
                # Extract loss and epoch
                losses.append(data['loss'])
                epochs.append(data['epoch'])
            except (json.JSONDecodeError, KeyError) as e:
                # Skip lines that can't be parsed
                continue

print(f"Found {len(losses)} training steps")
print(f"Epoch range: {min(epochs):.2f} to {max(epochs):.2f}")
print(f"Loss range: {min(losses):.4f} to {max(losses):.4f}")

# Create the plot
plt.figure(figsize=(12, 6))

# Plot loss vs step
plt.subplot(1, 2, 1)
plt.plot(range(len(losses)), losses, linewidth=1, alpha=0.7, color='blue')
plt.xlabel('Training Step', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.title('Training Loss vs Step', fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot loss vs epoch
plt.subplot(1, 2, 2)
plt.plot(epochs, losses, linewidth=1, alpha=0.7, color='green')
plt.xlabel('Epoch', fontweight='bold')
plt.ylabel('Loss', fontweight='bold')
plt.title('Training Loss vs Epoch', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved as 'training_loss_plot.png'")

# Optional: Show the plot
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Initial Loss: {losses[0]:.4f}")
print(f"Final Loss: {losses[-1]:.4f}")
print(f"Loss Reduction: {losses[0] - losses[-1]:.4f}")
print(f"Reduction %: {((losses[0] - losses[-1]) / losses[0]) * 100:.2f}%")
print(f"Min Loss: {min(losses):.4f} (at step {losses.index(min(losses))})")
print(f"Max Loss: {max(losses):.4f} (at step {losses.index(max(losses))})")
