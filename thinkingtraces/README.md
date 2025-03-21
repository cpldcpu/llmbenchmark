# LLM Thinking Trace Analyzer

This tool analyzes word statistics in chain-of-though traces of reasoning large language models. This is an attempt to compare the thinking process between different models.

## Background & Purpose

All investigated LLMs were prompted with the prompt below, and the (sometimes partial) reasoning traces were stores as a .txt file to be analyzed.

> You have two ropes, each of which takes exactly 60 minutes to burn completely. However, the ropes burn unevenly, meaning some parts may burn faster or slower than others. You have no other timing device. How can you measure exactly 20 minutes using these two ropes and matches to light them?

This specific prompt consistently generates long thinking traces across different models. It is an unsolvable logical puzzle, a fact which cannot be deduced logically by most of the current reasoning models.

By analyzing the first word of each line in these reasoning traces, we can identify relationships between different thinking models and potentially reveal insights about training methodologies.

## Output and Insights

![Word Frequency Heatmap Example](word_frequency_heatmap.png)

The heatmap colors indicate the frequency percentage of each word appearing as the first word in lines. Brighter colors represent higher percentages. 

By comparing these patterns, we can infer relationships between different models, their training processes and potentially identify when one the output of one model was used to finetune another.

The analysis of word statistics across various models reveals several interesting patterns:

- Certain words/tokens like "wait", "alternatively" often indicate backtracking in the thinking process
- These patterns emerge during reinforcement learning and can also be trained by finetuning on suitable thinking trace datasets.
- Some models show word statistics similar to DeepSeek R1, suggesting that R1 traces may have been used in their training process:
  - R1 thinking traces could have been used as finetuning data to "transfer" the thinking process
  - R1 traces might have "jumpstarted" the thinking ability with subsequent reinforcement learning to improve the reasoning process

Notable exceptions with different word statistics include: Sonnet, Gemini Flash and o3-mini. This suggests that these models have been trained with different methodologies or datasets. The absence of backtracking tokens like "Wait" or "Alternatively" in o3-mini models may indicate filtering of the output.

## Usage

Install dependencies with:
```bash
pip install pandas seaborn matplotlib
```

Run the analyzer on a folder containing text files with thinking traces:

```bash
python extract.py <folder_path>
```

For example:
```bash
python extract.py .
```

## Output

The tool generates several outputs:

1. **Text-based frequency histograms** for each file, showing the top 10 most common first words
2. **Comparison matrix** displaying the percentage frequency of common words across all files
3. **Heatmap visualization** saved as `word_frequency_heatmap.png`, providing a visual comparison of word frequencies

