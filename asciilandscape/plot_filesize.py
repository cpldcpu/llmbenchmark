import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_variance_folders(root_dir):
    """Find all folders matching the pattern 'variance_xxx'"""
    variance_folders = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if re.match(r'llama4_.*', dirname):
                variance_folders.append(os.path.join(dirpath, dirname))
    return variance_folders

def get_llm_name(folder_path):
    """Extract LLM name from variance folder name"""
    folder_name = os.path.basename(folder_path)
    return folder_name.replace('variance_', '')

def calculate_python_file_sizes(folder_path):
    """Calculate stats for Python file sizes in the folder"""
    file_sizes = []
    
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                file_size = os.path.getsize(file_path)
                file_sizes.append(file_size)
    
    if not file_sizes:
        return {
            'total_size': 0,
            'file_count': 0,
            'avg_size': 0,
            'stddev': 0
        }
    
    return {
        'total_size': sum(file_sizes),
        'file_count': len(file_sizes),
        'avg_size': np.mean(file_sizes),
        'stddev': np.std(file_sizes)
    }

def plot_file_sizes(llm_data):
    """Create a bar chart of average file sizes with stddev as error bars"""
    llms = list(llm_data.keys())
    avg_sizes = [data['avg_size'] / 1024 for data in llm_data.values()]  # Convert to KB
    stddevs = [data['stddev'] / 1024 for data in llm_data.values()]  # Convert to KB
    file_counts = [data['file_count'] for data in llm_data.values()]
    
    # Sort by average file size
    sorted_indices = np.argsort(avg_sizes)
    llms = [llms[i] for i in sorted_indices]
    avg_sizes = [avg_sizes[i] for i in sorted_indices]
    stddevs = [stddevs[i] for i in sorted_indices]
    file_counts = [file_counts[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bar chart with error bars
    bars = ax.bar(llms, avg_sizes, yerr=stddevs, capsize=5, color='royalblue', 
                  alpha=0.7, ecolor='black', error_kw={'elinewidth': 1.5})
    
    ax.set_xlabel('LLM', fontsize=12)
    ax.set_ylabel('Average Python File Size (KB)', fontsize=12)
    ax.set_title('Average Python File Size by LLM with Standard Deviation', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    # Add annotations for average size and file count
    for bar, avg, count in zip(bars, avg_sizes, file_counts):
        height = bar.get_height()
        ax.annotate(f'{avg:.1f} KB\n(n={count})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(os.path.dirname(os.path.abspath(__file__))) / "python_file_sizes_by_llm.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to: {output_path}")
    
    plt.show()

def main():
    # Start from the current directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Searching for variance folders in {root_dir}...")
    variance_folders = find_variance_folders(root_dir)
    
    if not variance_folders:
        print("No variance folders found.")
        return
    
    print(f"Found {len(variance_folders)} variance folders.")
    
    # Collect data for each LLM
    llm_data = {}
    for folder in variance_folders:
        llm_name = get_llm_name(folder)
        print(f"Processing {llm_name}...")
        size_data = calculate_python_file_sizes(folder)
        llm_data[llm_name] = size_data
        print(f"  - Found {size_data['file_count']} Python files")
        print(f"  - Total size: {size_data['total_size'] / 1024:.2f} KB")
        print(f"  - Average size: {size_data['avg_size'] / 1024:.2f} KB")
        print(f"  - Std Dev: {size_data['stddev'] / 1024:.2f} KB")
    
    # Plot the results
    plot_file_sizes(llm_data)

if __name__ == "__main__":
    main()
