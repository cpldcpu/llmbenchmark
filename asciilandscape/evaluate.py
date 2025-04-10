import os
import re
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import traceback

def find_llm_folders(root_dir):
    """Find all folders that contain iteration scripts"""
    llm_folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Check if folder contains iteration files
            has_iteration_files = any(
                re.match(r'iteration\d+\.py', filename)
                for filename in os.listdir(item_path)
            )
            if has_iteration_files:
                llm_folders.append(item_path)
    return llm_folders

def get_llm_name(folder_path):
    """Extract LLM name from folder name"""
    return os.path.basename(folder_path)

def find_iteration_scripts(folder_path):
    """Find all iteration scripts in a folder and sort them numerically"""
    iteration_scripts = []
    
    for filename in os.listdir(folder_path):
        match = re.match(r'iteration(\d+)\.py', filename)
        if match:
            script_path = os.path.join(folder_path, filename)
            iteration_number = int(match.group(1))
            iteration_scripts.append((iteration_number, script_path))
    
    # Sort by iteration number
    iteration_scripts.sort(key=lambda x: x[0])
    return iteration_scripts

def execute_script(script_path):
    """Execute a Python script and return its output and execution status"""
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30  # Set a timeout to prevent infinite loops
        )
        
        if result.returncode != 0:
            return {
                'status': 'error',
                'output': None,
                'error': result.stderr,
                'file_length': os.path.getsize(script_path),
                'output_length': 0
            }
        else:
            return {
                'status': 'success',
                'output': result.stdout,
                'error': None,
                'file_length': os.path.getsize(script_path),
                'output_length': len(result.stdout)
            }
    except subprocess.TimeoutExpired:
        return {
            'status': 'error',
            'output': None,
            'error': 'Execution timed out (30s)',
            'file_length': os.path.getsize(script_path),
            'output_length': 0
        }
    except Exception as e:
        return {
            'status': 'error',
            'output': None,
            'error': traceback.format_exc(),
            'file_length': os.path.getsize(script_path),
            'output_length': 0
        }

def plot_statistics(llm_data):
    """
    Plot statistics for all evaluated LLMs:
    1. Success rates
    2. File lengths
    3. Output lengths
    """
    llms = list(llm_data.keys())
    
    # Prepare data for plotting
    success_rates = []
    avg_file_lengths = []
    file_length_stddevs = []
    avg_output_lengths = []
    output_length_stddevs = []
    
    for llm, data in llm_data.items():
        total_scripts = len(data['results'])
        error_count = sum(1 for result in data['results'] if result['status'] == 'error')
        success_count = total_scripts - error_count
        success_rate = success_count / total_scripts if total_scripts > 0 else 0
        
        file_lengths = [result['file_length'] for result in data['results']]
        output_lengths = [result['output_length'] for result in data['results']] 
        
        success_rates.append(success_rate)
        avg_file_lengths.append(np.mean(file_lengths) / 1024)  # Convert to KB
        file_length_stddevs.append(np.std(file_lengths) / 1024)  # Convert to KB
        avg_output_lengths.append(np.mean(output_lengths))
        output_length_stddevs.append(np.std(output_lengths))
    
    # Sort all data by success rate (high to low)
    sorted_indices = np.argsort(success_rates)[::-1]  # Reversed to show highest first
    llms = [llms[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]
    avg_file_lengths = [avg_file_lengths[i] for i in sorted_indices]
    file_length_stddevs = [file_length_stddevs[i] for i in sorted_indices]
    avg_output_lengths = [avg_output_lengths[i] for i in sorted_indices]
    output_length_stddevs = [output_length_stddevs[i] for i in sorted_indices]
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Success rates
    bars1 = axs[0].bar(llms, success_rates, color='mediumseagreen')
    axs[0].set_xlabel('LLM')
    axs[0].set_ylabel('Success Rate')
    axs[0].set_title('Success Rate by LLM')
    axs[0].set_ylim(0, 1)
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add annotations for success rate
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        axs[0].annotate(f'{rate:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    # File lengths with error bars
    bars2 = axs[1].bar(llms, avg_file_lengths, yerr=file_length_stddevs, capsize=5,
                        color='royalblue', alpha=0.7, ecolor='black', 
                        error_kw={'elinewidth': 1.5})
    axs[1].set_xlabel('LLM')
    axs[1].set_ylabel('Average File Size (KB)')
    axs[1].set_title('Average File Size by LLM with Standard Deviation')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add annotations for file sizes
    for bar, avg in zip(bars2, avg_file_lengths):
        height = bar.get_height()
        axs[1].annotate(f'{avg:.1f} KB',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    # Output lengths with error bars
    bars3 = axs[2].bar(llms, avg_output_lengths, yerr=output_length_stddevs, capsize=5,
                        color='salmon', alpha=0.7, ecolor='black',
                        error_kw={'elinewidth': 1.5})
    axs[2].set_xlabel('LLM')
    axs[2].set_ylabel('Average Output Length (chars)')
    axs[2].set_title('Average Generated Output Length by LLM with Standard Deviation')
    axs[2].tick_params(axis='x', rotation=45)
    axs[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add annotations for output lengths
    for bar, avg in zip(bars3, avg_output_lengths):
        height = bar.get_height()
        axs[2].annotate(f'{avg:.0f} chars',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    plt.tight_layout()
    
    # Save the plots
    output_path = Path(os.path.dirname(os.path.abspath(__file__))) / "llm_evaluation_stats.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Statistics plot saved to: {output_path}")
    
    plt.show()

def generate_markdown_report(llm_data, output_path):
    """Generate a markdown report with the results of all scripts"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# ASCII Landscape Evaluation Results\n\n")
        
        for llm, data in llm_data.items():
            f.write(f"## {llm}\n\n")
            
            # Summary statistics
            total_scripts = len(data['results'])
            error_count = sum(1 for result in data['results'] if result['status'] == 'error')
            success_count = total_scripts - error_count
            
            f.write(f"- Total scripts: {total_scripts}\n")
            f.write(f"- Successful executions: {success_count}\n")
            f.write(f"- Failed executions: {error_count}\n")
            f.write(f"- Success rate: {success_count/total_scripts:.2%}\n\n")
            
            # Individual script results
            for i, result in enumerate(data['results']):
                iteration_num = data['iteration_numbers'][i]
                f.write(f"### Iteration {iteration_num}\n\n")
                
                if result['status'] == 'success':
                    f.write(f"- Status: ✅ Success\n")
                    f.write(f"- File length: {result['file_length']} bytes\n")
                    f.write(f"- Output length: {result['output_length']} characters\n\n")
                    f.write("#### Output:\n\n")
                    f.write("```\n")
                    f.write(result['output'])
                    f.write("```\n\n")
                else:
                    f.write(f"- Status: ❌ Error\n")
                    f.write(f"- File length: {result['file_length']} bytes\n")
                    f.write("#### Error message:\n\n")
                    f.write("```\n")
                    f.write(result['error'])
                    f.write("```\n\n")
            
            f.write("\n---\n\n")
        
        # Add the plot to the report
        f.write("## Evaluation Statistics\n\n")
        f.write("![Evaluation Statistics](llm_evaluation_stats.png)\n")
    
    print(f"Markdown report saved to: {output_path}")

def main():
    # Start from the current directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Searching for LLM folders in {root_dir}...")
    llm_folders = find_llm_folders(root_dir)
    
    if not llm_folders:
        print("No LLM folders found.")
        return
    
    print(f"Found {len(llm_folders)} LLM folders.")
    
    # Collect data for each LLM
    llm_data = {}
    
    for folder in llm_folders:
        llm_name = get_llm_name(folder)
        print(f"\nProcessing {llm_name}...")
        
        iteration_scripts = find_iteration_scripts(folder)
        print(f"  - Found {len(iteration_scripts)} iteration scripts")
        
        results = []
        iteration_numbers = []
        
        for iteration_num, script_path in iteration_scripts:
            print(f"  - Executing iteration{iteration_num}.py...")
            result = execute_script(script_path)
            results.append(result)
            iteration_numbers.append(iteration_num)
            
            status = "✅ Success" if result['status'] == 'success' else f"❌ Error: {result['error'].splitlines()[0] if result['error'] else 'Unknown error'}"
            print(f"    {status}")
        
        llm_data[llm_name] = {
            'results': results,
            'iteration_numbers': iteration_numbers
        }
    
    # Plot statistics
    plot_statistics(llm_data)
    
    # Generate markdown report
    results_path = Path(root_dir) / "results.md"
    generate_markdown_report(llm_data, results_path)

if __name__ == "__main__":
    main()