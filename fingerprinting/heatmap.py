import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import re
import unicodedata

def main(source_file, output_file):
    # Load and process the JSON data
    with open(source_file, 'r') as file:
        data = json.load(file)

    # Extract the prompt from the first entry (assuming all entries have the same prompt)
    prompt = data['results'][0]['prompt']
    print(f"Prompt: {prompt}\n")

    # Function to clean up responses
    def clean_response(response):
        normalized_response = unicodedata.normalize('NFKD', response)
        words = normalized_response.split()
        if not words:
            return ""
        first_word = words[0]
        # Remove non-alphanumeric characters and convert to lowercase
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', first_word.lower())
        return cleaned.strip()
    
    # Process the data
    processed_data = []
    for result in data['results']:
        llm = result['llm']
        for output in result['output']:
            cleaned_output = clean_response(output)
            if cleaned_output:  # Only add non-empty responses
                processed_data.append({'llm': llm, 'response': cleaned_output, 'original': output})

    # Create a DataFrame
    df = pd.DataFrame(processed_data)

    # Debugging: Print sample of raw and processed data
    print("Sample of raw and processed data:")
    print(df[['llm', 'original', 'response']].head(10))
    print("\n")

    # Debugging: Print unique values and their counts
    print("Unique LLMs:", df['llm'].unique())
    print("Unique responses:", df['response'].unique())
    print("\nResponse value counts:")
    print(df['response'].value_counts())
    print("\n")

    # Create a pivot table for the heatmap
    pivot_table = pd.pivot_table(df, values='response', index='llm', columns='response', aggfunc='count', fill_value=0)

    # Debugging: Print pivot table shape and contents
    print("Pivot table shape:", pivot_table.shape)
    print("Pivot table contents:")
    print(pivot_table)
    print("\n")

    # Check if the pivot table is empty
    if pivot_table.empty:
        print("Error: The pivot table is empty. No heatmap can be generated.")
    else:
        # Create the heatmap
        plt.figure(figsize=(20, 12))  # Increased figure size for better text visibility
        ax = sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='d', cbar_kws={'label': 'Frequency'})
        
        # Increase font size of annotation (the numbers in each cell)
        for t in ax.texts:
            t.set_size(20)
       
        plt.title(f'LLM Response Heatmap\nPrompt: "{prompt}"', fontsize=20, wrap=True)        
        # plt.title('LLM Response Heatmap', fontsize=24)
        plt.xlabel('Response', fontsize=22)
        plt.ylabel('LLM', fontsize=22)
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.yticks(fontsize=18)
        cbar = plt.gcf().axes[-1]
        cbar.tick_params(labelsize=14)
        cbar.set_ylabel('Frequency', fontsize=18)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap has been generated and saved as '{output_file}'")

    print("Script execution completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    This script generates a heatmap from LLM (Language Model) response data stored in a JSON file.
    It processes the data, cleans the responses, and creates a visual representation of response
    frequencies across different LLMs. The heatmap shows LLMs on the y-axis, responses on the x-axis,
    and uses color intensity to represent the frequency of each response for each LLM.

    The script performs the following steps:
    1. Loads and processes the JSON data
    2. Cleans and standardizes the responses
    3. Creates a pivot table of the data
    4. Generates a heatmap using seaborn and matplotlib
    5. Saves the heatmap as a PNG file

    The output includes debugging information and the final heatmap image.
    """)
    parser.add_argument('--sourcename', type=str, default='output_queries.json',
                        help='Name of the source JSON file (default: output_queries.json)')
    parser.add_argument('--output', type=str, default='llm_response_heatmap.png',
                        help='Name of the output image file (default: llm_response_heatmap.png)')
    args = parser.parse_args()

    main(args.sourcename, args.output)