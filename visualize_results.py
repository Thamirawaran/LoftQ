"""
Visualize LoftQ Test Results
Creates charts and graphs from test results
"""
import json
import matplotlib.pyplot as plt
import os

# Load test results
with open('test_results/loftq_test_results_20251205_235643.json', 'r') as f:
    results = json.load(f)

# Create visualizations directory
os.makedirs('test_results/visualizations', exist_ok=True)

# Extract efficiency data
efficiency_data = None
for test in results['tests']:
    if 'efficiency_data' in test:
        efficiency_data = test['efficiency_data']
        break

if efficiency_data:
    ranks = [d['rank'] for d in efficiency_data]
    trainable_params = [d['trainable_params'] for d in efficiency_data]
    reduction_factors = [d['reduction_factor'] for d in efficiency_data]
    memory_savings = [d['memory_savings_percent'] for d in efficiency_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LoftQ Test Results - Parameter Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Trainable Parameters vs Rank
    axes[0, 0].bar(ranks, trainable_params, color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel('LoRA Rank', fontweight='bold')
    axes[0, 0].set_ylabel('Trainable Parameters', fontweight='bold')
    axes[0, 0].set_title('Trainable Parameters by LoRA Rank')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, (r, p) in enumerate(zip(ranks, trainable_params)):
        axes[0, 0].text(r, p, f'{p:,}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Reduction Factor vs Rank
    axes[0, 1].plot(ranks, reduction_factors, marker='o', linewidth=2, markersize=8, color='green')
    axes[0, 1].set_xlabel('LoRA Rank', fontweight='bold')
    axes[0, 1].set_ylabel('Reduction Factor (x)', fontweight='bold')
    axes[0, 1].set_title('Parameter Reduction Factor by LoRA Rank')
    axes[0, 1].grid(True, alpha=0.3)
    for r, rf in zip(ranks, reduction_factors):
        axes[0, 1].text(r, rf, f'{rf}x', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Memory Savings Percentage
    axes[1, 0].bar(ranks, memory_savings, color='coral', alpha=0.7)
    axes[1, 0].set_xlabel('LoRA Rank', fontweight='bold')
    axes[1, 0].set_ylabel('Memory Savings (%)', fontweight='bold')
    axes[1, 0].set_title('Memory Savings Percentage by LoRA Rank')
    axes[1, 0].set_ylim([90, 100])
    axes[1, 0].grid(axis='y', alpha=0.3)
    for r, ms in zip(ranks, memory_savings):
        axes[1, 0].text(r, ms, f'{ms}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Comparison Table
    axes[1, 1].axis('off')
    table_data = []
    table_data.append(['Rank', 'Trainable\nParams', 'Reduction\nFactor', 'Memory\nSavings'])
    for d in efficiency_data:
        table_data.append([
            str(d['rank']),
            f"{d['trainable_params']:,}",
            f"{d['reduction_factor']}x",
            f"{d['memory_savings_percent']}%"
        ])
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                            colWidths=[0.15, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    axes[1, 1].set_title('Efficiency Comparison Table', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('test_results/visualizations/parameter_efficiency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: test_results/visualizations/parameter_efficiency.png")

# Generation time visualization
gen_test = None
for test in results['tests']:
    if test['name'] == 'Text Generation' and test['status'] == 'PASSED':
        gen_test = test
        break

if gen_test:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prompts = [g['prompt'][:30] + '...' if len(g['prompt']) > 30 else g['prompt'] 
               for g in gen_test['generations']]
    times = [g['generation_time_seconds'] for g in gen_test['generations']]
    
    bars = ax.barh(prompts, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    ax.set_xlabel('Generation Time (seconds)', fontweight='bold')
    ax.set_title('Text Generation Performance by Prompt', fontweight='bold', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(time, bar.get_y() + bar.get_height()/2, f'{time}s',
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Add average line
    avg_time = gen_test['average_generation_time']
    ax.axvline(x=avg_time, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_time}s')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('test_results/visualizations/generation_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: test_results/visualizations/generation_performance.png")

# Test summary pie chart
fig, ax = plt.subplots(figsize=(8, 8))
passed = results['summary']['passed']
failed = results['summary']['failed']

colors = ['#4CAF50', '#F44336']
explode = (0.1, 0)
sizes = [passed, failed] if failed > 0 else [passed]
labels = [f'Passed\n({passed})', f'Failed\n({failed})'] if failed > 0 else [f'Passed\n({passed})']
colors_used = colors[:len(sizes)]

ax.pie(sizes, explode=explode[:len(sizes)], labels=labels, colors=colors_used,
       autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title(f'Test Results Summary\n{passed}/{results["summary"]["total_tests"]} Tests Passed',
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('test_results/visualizations/test_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: test_results/visualizations/test_summary.png")

print("\n" + "="*60)
print("All visualizations created successfully!")
print("="*60)
print("\nVisualization files:")
print("1. parameter_efficiency.png - Parameter efficiency analysis")
print("2. generation_performance.png - Text generation performance")
print("3. test_summary.png - Overall test results summary")
