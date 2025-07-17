# Text Subcategory Classification

This project automatically classifies text documents into subcategories using advanced NLP techniques with transformers.

## Problem Statement

Given text documents organized into main categories (business, entertainment, sport), the goal is to further classify them into specific subcategories:

- **Business**: stock market, company news, mergers and acquisitions, etc.
- **Entertainment**: cinema, theatre, music, literature, personality, etc.
- **Sport**: cricket, football, Olympics, etc.

## Solution Approach

The solution uses **zero-shot classification** with BART (Bidirectional and Auto-Regressive Transformers) to classify texts without requiring training data. This approach is:

- **Simple**: No training required
- **Flexible**: Easy to add new subcategories
- **Effective**: Uses state-of-the-art transformer model
- **Fast**: Direct inference without fine-tuning

## Repository Structure

```
nlp-text-classification/
├── data/                          # Your text data
│   ├── business/                  # Business category texts
│   │   ├── 001.txt
│   │   ├── 002.txt
│   │   └── ...
│   ├── entertainment/             # Entertainment category texts
│   │   ├── 001.txt
│   │   ├── 002.txt
│   │   └── ...
│   └── sport/                     # Sport category texts
│       ├── 001.txt
│       ├── 002.txt
│       └── ...
├── results/                       # Output directory
│   ├── classification_results.json
│   ├── classification_results.csv
│   └── summary_statistics.csv
├── main.py                        # Main classification script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd nlp-text-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Create the `data/` folder in the project root
   - Add your text files in the respective subfolders:
     - `data/business/`
     - `data/entertainment/`
     - `data/sport/`

## Usage

Run the classification:

```bash
python main.py
```

The script will:
1. Load all `.txt` files from the data folders
2. Classify each text into appropriate subcategories
3. Save results in the `results/` folder

## Output

The script generates three output files:

1. **`classification_results.json`**: Detailed results with confidence scores
2. **`classification_results.csv`**: Tabular format for easy viewing
3. **`summary_statistics.csv`**: Count of documents per subcategory

## Example Output

```
Processing business category...
Classifying business: 100%|██████████| 50/50 [00:45<00:00,  1.10it/s]

Processing entertainment category...
Classifying entertainment: 100%|██████████| 30/30 [00:28<00:00,  1.07it/s]

Processing sport category...
Classifying sport: 100%|██████████| 25/25 [00:23<00:00,  1.08it/s]

Results saved to results
Total files processed: 105

Summary by category:
    category     subcategory  count
0   business    stock market     15
1   business  company earnings     12
2   business   corporate news      8
...
```

## Technical Details

- **Model**: facebook/bart-large-mnli
- **Technique**: Zero-shot classification
- **Framework**: Hugging Face Transformers
- **Input**: Raw text files
- **Output**: JSON/CSV with subcategory predictions and confidence scores

## Customization

To modify subcategories, edit the `subcategories` dictionary in `main.py`:

```python
self.subcategories = {
    'business': ['your', 'custom', 'subcategories'],
    # ...
}
```

## Performance

- Processing speed: ~1-2 texts per second
- Memory usage: ~2-4GB RAM (depending on model)
- Accuracy: High for clear, well-written texts

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- 4GB+ RAM recommended