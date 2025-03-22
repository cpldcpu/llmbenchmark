import sys
import os
from collections import Counter
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def clean_word(word):
    """Remove punctuation and convert to lowercase."""
    return re.sub(r'[,.!?]', '', word.lower())

def process_file(filepath):
    """Process a single file and return word histogram."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        # Get first words from non-empty lines, ignoring numbers and list symbols
        first_words = [clean_word(re.sub(r'^[\d\.\-\*\s]+', '', line).split()[0]) 
                      for line in text.splitlines() 
                      if line.strip() and re.sub(r'^[\d\.\-\*\s]+', '', line).split()]
        total_words = len(first_words)
        counts = Counter(first_words)
        # Convert to percentages
        percentages = {word: (count/total_words*100) 
                      for word, count in counts.items()}
        return percentages

def print_histograms(file_histograms):
    """Print histograms for all files as text and create comparison matrix."""
    # Get all unique words from top 10 of each file
    top_words = set()
    file_tops = {}
    
    for filename, histogram in file_histograms.items():
        # Store top 10 words and percentages for each file, sorted by percentage
        file_tops[filename] = dict(sorted(histogram.items(), 
                                        key=lambda x: (-x[1], x[0]))[:10])
        top_words.update(file_tops[filename].keys())
    
    # Print individual histograms
    for filename, hist in sorted(file_tops.items()):
        print(f"\nWord Frequency: {Path(filename).stem}")
        print("-" * 40)
        # Sort by percentage descending
        for word, pct in sorted(hist.items(), key=lambda x: (-x[1], x[0])):
            print(f"{word:15} {pct:6.2f}%")
    
    # Create comparison matrix with words as rows and files as columns
    matrix_data = {}
    for word in sorted(top_words):
        row = {}
        for filename in sorted(file_tops.keys()):
            if word in file_tops[filename]:
                row[Path(filename).stem] = file_tops[filename][word]
        matrix_data[word] = row
    
    # Create DataFrame with words as index, sorted by maximum percentage
    df = pd.DataFrame.from_dict(matrix_data, orient='index')
    df = df.reindex(df.max(axis=1).sort_values(ascending=False).index)
    
    # Take only top 20 rows
    df = df.head(20)

    # sort
    df = df[sorted(df.columns)]
    
    print("\nComparison Matrix (percentages):")
    print("-" * 40)
    print(df.round(2).to_string(na_rep=''))
    
    # Create a more compact heatmap visualization
    # Create figure with tighter sizing
    plt.figure(figsize=(10, 6))
    
    # Create mask for NaN values to leave them white
    mask = df.isna()
    
    # Create the heatmap with improved styling for compactness
    ax = sns.heatmap(
        df,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Percentage', 'shrink': 0.6, 'aspect': 10},
        linewidths=0.2,
        mask=mask,
        annot_kws={"size": 9, "weight": "bold"},
        square=False  # Allow rectangular cells for better space usage
    )
    
    # Improve title and labels with more compact styling
    plt.title('Word Frequency Heatmap of First Word per Line', fontsize=12, fontweight='bold', pad=10)
    plt.ylabel('Words', fontsize=10, fontweight='bold', labelpad=5)
    plt.xlabel('Files', fontsize=10, fontweight='bold', labelpad=5)
    
    # Fix overlapping labels with more compact placement
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=8)
    
    # Move x-axis labels to the top
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    
    # Reduce padding around the plot
    plt.tight_layout(pad=1.0)
    
    # Save the heatmap to a file before displaying
    output_path = Path(os.path.dirname(os.path.abspath(__file__))) / "word_frequency_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract.py <folder_path>")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)

    # Process all txt files
    histograms = {}
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            filepath = os.path.join(folder_path, file)
            histograms[file] = process_file(filepath)

    if not histograms:
        print("No .txt files found in the specified directory")
        sys.exit(1)

    # Print results
    print_histograms(histograms)

if __name__ == "__main__":
    main()
