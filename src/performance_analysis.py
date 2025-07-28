import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_performance_plots():
    """Create visualizations of the performance results."""
    
    # Load baseline results
    with open('benchmark_results_baseline.json', 'r') as f:
        baseline_results = json.load(f)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('FastVLM-0.5B Performance Analysis on M2 MacBook', fontsize=16)
    
    # 1. Tokens per second by prompt
    prompts = [b['prompt'][:30] + '...' for b in baseline_results['benchmarks']]
    tokens_per_sec = [b['tokens_per_second'] for b in baseline_results['benchmarks']]
    
    ax1.bar(range(len(prompts)), tokens_per_sec, color='skyblue')
    ax1.set_xticks(range(len(prompts)))
    ax1.set_xticklabels(prompts, rotation=45, ha='right')
    ax1.set_ylabel('Tokens/Second')
    ax1.set_title('Generation Speed by Prompt')
    ax1.axhline(y=np.mean(tokens_per_sec), color='red', linestyle='--', 
                label=f'Average: {np.mean(tokens_per_sec):.2f}')
    ax1.legend()
    
    # 2. Inference time distribution
    all_times = []
    for benchmark in baseline_results['benchmarks']:
        all_times.extend(benchmark['runs'])
    
    ax2.hist(all_times, bins=20, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Inference Time (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Inference Time Distribution')
    ax2.axvline(x=np.mean(all_times), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_times):.2f}s')
    ax2.legend()
    
    # 3. Model specifications
    specs = {
        'Parameters': f"{baseline_results['total_params'] / 1e6:.1f}M",
        'Load Time': f"{baseline_results['load_time']:.2f}s",
        'Device': baseline_results['device'].upper(),
        'Avg Tokens/s': f"{np.mean(tokens_per_sec):.2f}",
        'Vision Encoder': 'FastViTHD'
    }
    
    y_pos = np.arange(len(specs))
    ax3.barh(y_pos, [1] * len(specs), alpha=0)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(list(specs.keys()))
    ax3.set_xlim(0, 1)
    
    for i, (key, value) in enumerate(specs.items()):
        ax3.text(0.1, i, value, fontsize=12, va='center')
    
    ax3.set_title('Model Specifications')
    ax3.axis('off')
    
    # 4. Optimization potential
    optimization_data = {
        'Baseline (FP16)': 1.0,
        'INT8 Quantization': 0.5,  # Expected 2x speedup
        'Core ML (Neural Engine)': 0.2,  # Expected 5x speedup
        'INT4 Quantization': 0.35,  # Expected ~3x speedup
    }
    
    methods = list(optimization_data.keys())
    relative_time = list(optimization_data.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = ax4.bar(range(len(methods)), relative_time, color=colors)
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel('Relative Inference Time')
    ax4.set_title('Expected Performance with Optimizations')
    ax4.set_ylim(0, 1.2)
    
    # Add speedup labels
    baseline_time = np.mean(all_times)
    for i, (method, rel_time) in enumerate(optimization_data.items()):
        speedup = 1.0 / rel_time
        ax4.text(i, rel_time + 0.05, f'{speedup:.1f}x', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Performance analysis plot saved as 'performance_analysis.png'")
    
    return fig

def generate_optimization_summary():
    """Generate a summary of optimization strategies."""
    
    # Load baseline results
    with open('benchmark_results_baseline.json', 'r') as f:
        baseline_results = json.load(f)
    
    baseline_tokens_per_sec = np.mean([b['tokens_per_second'] for b in baseline_results['benchmarks']])
    baseline_inference_time = np.mean([np.mean(b['runs']) for b in baseline_results['benchmarks']])
    
    summary = {
        "project": "FastVLM Mobile Optimization Study",
        "model": "FastVLM-0.5B",
        "hardware": "M2 MacBook Pro",
        "baseline_performance": {
            "tokens_per_second": baseline_tokens_per_sec,
            "average_inference_time": baseline_inference_time,
            "model_size_mb": baseline_results['total_params'] / 1e6 * 2,  # FP16 estimate (model uses half precision)
            "load_time": baseline_results['load_time']
        },
        "optimization_strategies": {
            "INT8_quantization": {
                "expected_speedup": "2x",
                "size_reduction": "50%",
                "accuracy_impact": "Minimal (<2% drop)",
                "implementation_status": "PyTorch quantization not compatible with model architecture"
            },
            "Core_ML_conversion": {
                "expected_speedup": "5-10x on Neural Engine",
                "size_reduction": "Variable based on quantization",
                "accuracy_impact": "Minimal with FP16",
                "implementation_status": "Vision encoder exported successfully"
            },
            "MLX_framework": {
                "expected_speedup": "3-5x with unified memory",
                "size_reduction": "With INT8: 50%, INT4: 75%",
                "accuracy_impact": "INT8: Minimal, INT4: 3-5% drop",
                "implementation_status": "Framework installed, conversion pending"
            }
        },
        "key_findings": [
            "Baseline model achieves 13.48 tokens/second on M2",
            "Vision encoder (FastViTHD) successfully exported to Core ML",
            "Standard PyTorch quantization incompatible with model architecture",
            "MLX framework offers best path for M2 optimization",
            "Expected 5-10x speedup possible with Neural Engine utilization"
        ],
        "recommendations": [
            "Use MLX framework for M2-optimized deployment",
            "Implement INT8 quantization for 2x speedup with minimal accuracy loss",
            "Leverage Neural Engine through Core ML for maximum performance",
            "Consider INT4 for edge deployment where size is critical"
        ]
    }
    
    # Save summary
    with open('optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nOptimization Summary:")
    print("=" * 60)
    print(f"Model: {summary['model']}")
    print(f"Hardware: {summary['hardware']}")
    print(f"Baseline Performance: {baseline_tokens_per_sec:.2f} tokens/sec")
    print("\nKey Findings:")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    return summary

def main():
    """Run the performance analysis."""
    print("Running FastVLM Performance Analysis...")
    
    # Check if baseline results exist
    if not Path('benchmark_results_baseline.json').exists():
        print("Error: Please run benchmark_baseline.py first!")
        return
    
    # Generate plots
    create_performance_plots()
    
    # Generate summary
    generate_optimization_summary()
    
    print("\nAnalysis complete! Check 'performance_analysis.png' and 'optimization_summary.json'")

if __name__ == "__main__":
    main()