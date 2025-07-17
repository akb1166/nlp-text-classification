# main.py
import os
import json
from pathlib import Path
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class TextSubcategoryClassifier:
    def __init__(self):
        """Initialize the classifier with zero-shot capabilities"""
        # Load the zero-shot classification pipeline
        print("Loading classifier...")
        try:
            self.classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",  
                # device=-1  # Force CPU to avoid GPU issues
            )
            print("✅ Classifier loaded successfully")
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            raise
        
        # Define subcategories for each main category
        self.subcategories = {
            'business': [
                'earnings and financial reports',
                'stock market trading', 
                'mergers and acquisitions',
                'corporate news and strategy',
                'economic analysis'
            ],
            'entertainment': [
                'movies and cinema',
                'television and streaming',
                'music and concerts',
                'celebrity news',
                'arts and culture'
            ],
            'sport': [
                'football and NFL',
                'basketball and NBA',
                'baseball and MLB',
                'tennis and individual sports',
                'olympics and international sports'
            ]
        }

    
    def read_text_file(self, filepath):
        """Read text from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""
    
    def classify_text(self, text, category):
        """Classify text into subcategory"""
        if category not in self.subcategories:
            return "unknown"
        
        # Get relevant subcategories
        candidate_labels = self.subcategories[category]
        
        # Truncate text if too long (BART has token limits)
        text = text[:1000] if len(text) > 1000 else text
        
        # Perform zero-shot classification
        result = self.classifier(text, candidate_labels)
        
        # Return top prediction
        return {
            'subcategory': result['labels'][0],
            'confidence': result['scores'][0],
            'all_scores': dict(zip(result['labels'], result['scores']))
        }
    
    def process_directory(self, data_dir):
        """Process all text files in the data directory"""
        results = []
        
        for category in ['business', 'entertainment', 'sport']:
            category_path = Path(data_dir) / category
            
            if not category_path.exists():
                print(f"Warning: {category_path} does not exist")
                continue
            
            print(f"\nProcessing {category} category...")
            
            # Get all .txt files
            txt_files = list(category_path.glob("*.txt"))
            
            for txt_file in tqdm(txt_files, desc=f"Classifying {category}"):
                text = self.read_text_file(txt_file)
                
                if text:
                    classification = self.classify_text(text, category)
                    
                    results.append({
                        'file': str(txt_file),
                        'category': category,
                        'subcategory': classification['subcategory'],
                        'confidence': classification['confidence'],
                        'text_preview': text[:200] + "..." if len(text) > 200 else text
                    })
        
        return results
    
    def save_results(self, results, output_dir):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as JSON
        with open(output_path / 'classification_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV for easier viewing
        df = pd.DataFrame(results)
        df.to_csv(output_path / 'classification_results.csv', index=False)
        
        # Generate summary statistics
        summary = df.groupby(['category', 'subcategory']).size().reset_index(name='count')
        summary.to_csv(output_path / 'summary_statistics.csv', index=False)
        
        print(f"\nResults saved to {output_path}")
        print(f"Total files processed: {len(results)}")
        print("\nSummary by category:")
        print(summary.to_string(index=False))

def main():
    # Initialize classifier
    classifier = TextSubcategoryClassifier()
    
    # Set paths
    data_dir = "data"  # Your data folder with business, entertainment, sport subfolders
    output_dir = "results"
    
    # Process all files
    results = classifier.process_directory(data_dir)
    
    # Save results
    classifier.save_results(results, output_dir)

if __name__ == "__main__":
    main()