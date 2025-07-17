# utils.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_results(results_path="results/classification_results.json"):
    """Analyze and visualize classification results"""
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    print("=== CLASSIFICATION ANALYSIS ===\n")
    
    # Basic statistics
    print("1. Basic Statistics:")
    print(f"   Total documents: {len(df)}")
    print(f"   Categories: {df['category'].nunique()}")
    print(f"   Subcategories: {df['subcategory'].nunique()}")
    print(f"   Average confidence: {df['confidence'].mean():.3f}")
    
    # Distribution by category
    print("\n2. Distribution by Main Category:")
    category_counts = df['category'].value_counts()
    for cat, count in category_counts.items():
        print(f"   {cat}: {count} documents")
    
    # Top subcategories
    print("\n3. Top Subcategories:")
    subcat_counts = df['subcategory'].value_counts().head(10)
    for subcat, count in subcat_counts.items():
        print(f"   {subcat}: {count} documents")
    
    # Confidence analysis
    print("\n4. Confidence Analysis:")
    print(f"   High confidence (>0.8): {(df['confidence'] > 0.8).sum()} documents")
    print(f"   Medium confidence (0.5-0.8): {((df['confidence'] >= 0.5) & (df['confidence'] <= 0.8)).sum()} documents")
    print(f"   Low confidence (<0.5): {(df['confidence'] < 0.5).sum()} documents")
    
    return df

def create_visualizations(df, output_dir="results"):
    """Create visualization charts"""
    output_path = Path(output_dir)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribution by category
    plt.figure(figsize=(10, 6))
    df['category'].value_counts().plot(kind='bar')
    plt.title('Document Distribution by Main Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Documents')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top subcategories
    plt.figure(figsize=(12, 8))
    top_subcats = df['subcategory'].value_counts().head(15)
    top_subcats.plot(kind='barh')
    plt.title('Top 15 Subcategories')
    plt.xlabel('Number of Documents')
    plt.tight_layout()
    plt.savefig(output_path / 'top_subcategories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black')
    plt.title('Classification Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Documents')
    plt.axvline(df['confidence'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["confidence"].mean():.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_path}/")

def export_filtered_results(df, min_confidence=0.7, output_dir="results"):
    """Export high-confidence results"""
    high_conf = df[df['confidence'] >= min_confidence]
    
    output_path = Path(output_dir)
    high_conf.to_csv(output_path / f'high_confidence_results_{min_confidence}.csv', index=False)
    
    print(f"High confidence results (>={min_confidence}) saved: {len(high_conf)} documents")

def main():
    """Run analysis utilities"""
    print("Running analysis on classification results...\n")
    
    # Analyze results
    df = analyze_results()
    
    # Create visualizations (optional - requires matplotlib/seaborn)
    try:
        create_visualizations(df)
    except ImportError:
        print("\nSkipping visualizations (matplotlib/seaborn not installed)")
    
    # Export high-confidence results
    export_filtered_results(df, min_confidence=0.8)

if __name__ == "__main__":
    main()