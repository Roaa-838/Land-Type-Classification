
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EuroSATDataPipeline:
    """
    Complete data pipeline for EuroSAT Land Type Classification project.
    
    This class handles:
    - Data validation and cleaning
    - CSV processing and verification
    - Image loading and preprocessing
    - Label mapping
    - Data splitting and organization
    """
    
    def __init__(self, raw_data_dir: str = "data/raw/EuroSAT", 
                 processed_data_dir: str = "data/processed",
                 interim_data_dir: str = "data/interim"):
        """
        Initialize the data pipeline.
        
        Args:
            raw_data_dir: Path to raw EuroSAT data
            processed_data_dir: Path to save processed data
            interim_data_dir: Path to save intermediate data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.interim_data_dir = Path(interim_data_dir)
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.interim_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define land cover classes
        self.classes = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
            "Industrial", "Pasture", "PermanentCrop", "Residential", 
            "River", "SeaLake"
        ]
        
        self.label_map = {}
        self.class_counts = {}
        
    def validate_dataset_structure(self) -> Dict[str, int]:
        """
        Validate the dataset structure and count images per class.
        
        Returns:
            Dictionary with class names and their image counts
        """
        print("Validating dataset structure...")
        
        for cls in self.classes:
            cls_folder = self.raw_data_dir / cls
            
            if not cls_folder.exists():
                print(f"Missing folder: {cls_folder}")
                self.class_counts[cls] = 0
                continue
            
            # Count image files (JPG, PNG, TIF)
            image_files = list(cls_folder.glob("*.jpg")) + \
                         list(cls_folder.glob("*.png")) + \
                         list(cls_folder.glob("*.tif"))
            
            n_images = len(image_files)
            self.class_counts[cls] = n_images
            
            print(f"{cls:<20}: {n_images:>6} images")
        
        total_images = sum(self.class_counts.values())
        print(f"Total images: {total_images}")
        
        return self.class_counts
    
    def load_label_map(self) -> Dict[str, int]:
        """
        Load label mapping from label_map.json.
        
        Returns:
            Dictionary mapping class names to numeric labels
        """
        label_map_path = self.raw_data_dir / "label_map.json"
        
        if label_map_path.exists():
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            print("Label map loaded from label_map.json")
        else:
            # Create label map if it doesn't exist
            self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}
            print("label_map.json not found, creating default mapping")
            
            # Save the created label map
            with open(self.processed_data_dir / "label_map.json", 'w') as f:
                json.dump(self.label_map, f, indent=2)
        
        print("Label mapping:")
        for cls, label in self.label_map.items():
            print(f"   {cls:<20}: {label}")
        
        return self.label_map
    
    def clean_csv_files(self) -> Dict[str, pd.DataFrame]:
        """
        Clean and validate CSV files (train.csv, test.csv, validation.csv).
        
        Returns:
            Dictionary containing cleaned DataFrames
        """
        print("\\nCleaning CSV files...")
        
        cleaned_dfs = {}
        csv_files = ["train.csv", "test.csv", "validation.csv"]
        
        for csv_file in csv_files:
            csv_path = self.raw_data_dir / csv_file
            
            if not csv_path.exists():
                print(f"{csv_file} not found in {self.raw_data_dir}")
                continue
            
            print(f"\\nProcessing {csv_file}...")
            
            # Load CSV
            df = pd.read_csv(csv_path)
            print(f"Original shape: {df.shape}")
            
            # Remove index column if it exists
            index_cols = [col for col in df.columns if 
                         col.lower() in ['index', 'unnamed: 0', 'unnamed:0']]
            if index_cols:
                df = df.drop(columns=index_cols)
                print(f"Removed index column: {index_cols}")
            
            # Standardize column names
            if 'Filename' in df.columns:
                df = df.rename(columns={'Filename': 'filename'})
            if 'Label' in df.columns:
                df = df.rename(columns={'Label': 'label'})
            if 'ClassName' in df.columns:
                df = df.rename(columns={'ClassName': 'class_name'})
            
            # Verify file existence
            missing_files = []
            for idx, row in df.iterrows():
                if 'filename' in df.columns:
                    # Handle both relative and absolute paths
                    if str(row['filename']).startswith(str(self.raw_data_dir)):
                        img_path = Path(row['filename'])
                    else:
                        img_path = self.raw_data_dir / row['filename']
                    
                    if not img_path.exists():
                        missing_files.append(str(img_path))
            
            if missing_files:
                print(f"{len(missing_files)} missing images found")
                # Remove rows with missing images
                missing_names = [str(Path(f).name) for f in missing_files]
                df = df[~df['filename'].isin(missing_names)]
                print(f"Removed {len(missing_files)} rows with missing images")
            else:
                print("All images exist in dataset")
            
            # Add full file paths
            if 'filename' in df.columns:
                df['full_path'] = df['filename'].apply(
                    lambda x: str(self.raw_data_dir / x) if not str(x).startswith(str(self.raw_data_dir)) 
                    else x
                )
            
            print(f"Final shape: {df.shape}")
            
            # Save cleaned CSV
            save_path = self.processed_data_dir / csv_file
            df.to_csv(save_path, index=False)
            print(f"Saved to: {save_path}")
            
            cleaned_dfs[csv_file.replace('.csv', '')] = df
        
        return cleaned_dfs
    
    def load_and_preprocess_image(self, img_path: str, target_size: Tuple[int, int] = (64, 64), 
                                normalize: bool = True) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image.
        
        Args:
            img_path: Path to the image file
            target_size: Target size for resizing (height, width)
            normalize: Whether to normalize pixel values to [0, 1]
            
        Returns:
            Preprocessed image array or None if loading fails
        """
        try:
            img_path = Path(img_path)
            
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Load RGB image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img)
                
            elif img_path.suffix.lower() == '.tif':
                # Load TIF image (potentially multi-band)
                img_array = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img_array is None:
                    return None
                
                # Handle multi-band TIF files
                if len(img_array.shape) == 3:
                    img_array = cv2.resize(img_array, target_size)
                else:
                    img_array = cv2.resize(img_array, target_size)
                    img_array = np.stack([img_array] * 3, axis=-1)  # Convert to 3-channel
            else:
                print(f"Unsupported image format: {img_path.suffix}")
                return None
            
            # Normalize if requested
            if normalize:
                img_array = img_array.astype(np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None
    
    def generate_data_statistics(self, cleaned_dfs: Dict[str, pd.DataFrame]) -> None:
        """
        Generate and save data statistics and visualizations.
        
        Args:
            cleaned_dfs: Dictionary of cleaned DataFrames
        """
        print("\\n Generating data statistics...")
        print("=" * 50)
        
        # Create statistics summary
        stats = {
            'dataset_overview': {
                'total_classes': len(self.classes),
                'class_names': self.classes,
                'images_per_class': self.class_counts
            },
            'split_statistics': {}
        }
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Images per class
        plt.subplot(2, 2, 1)
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        plt.bar(classes, counts, color='skyblue', alpha=0.7)
        plt.title('Images per Class (Total Dataset)')
        plt.xlabel('Land Cover Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')
        
        # Plot 2: Split distribution
        plt.subplot(2, 2, 2)
        split_sizes = {split: len(df) for split, df in cleaned_dfs.items()}
        if split_sizes:
            plt.pie(split_sizes.values(), labels=split_sizes.keys(), autopct='%1.1f%%')
            plt.title('Dataset Split Distribution')
        
        # Plot 3: Class distribution per split
        if len(cleaned_dfs) > 0:
            plt.subplot(2, 2, (3, 4))
            split_class_data = []
            
            for split_name, df in cleaned_dfs.items():
                if 'class_name' in df.columns:
                    class_dist = df['class_name'].value_counts()
                    stats['split_statistics'][split_name] = {
                        'total_samples': len(df),
                        'class_distribution': class_dist.to_dict()
                    }
                    
                    for cls in self.classes:
                        count = class_dist.get(cls, 0)
                        split_class_data.append({
                            'Split': split_name,
                            'Class': cls,
                            'Count': count
                        })
            
            if split_class_data:
                split_df = pd.DataFrame(split_class_data)
                pivot_df = split_df.pivot(index='Class', columns='Split', values='Count').fillna(0)
                
                sns.heatmap(pivot_df, annot=True, fmt='g', cmap='Blues')
                plt.title('Class Distribution Across Splits')
                plt.ylabel('Land Cover Class')
                plt.xlabel('Dataset Split')
        
        plt.tight_layout()
        plt.savefig(self.processed_data_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save statistics as JSON
        with open(self.processed_data_dir / 'dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {self.processed_data_dir}")
    
    def create_sample_visualization(self, cleaned_dfs: Dict[str, pd.DataFrame], 
                                  n_samples: int = 2) -> None:
        """
        Create a visualization showing sample images from each class.
        
        Args:
            cleaned_dfs: Dictionary of cleaned DataFrames
            n_samples: Number of sample images per class to display
        """
        print(f"\\nCreating sample visualization ({n_samples} per class)...")
        
        if 'train' not in cleaned_dfs:
            print("No training data available for visualization")
            return
        
        train_df = cleaned_dfs['train']
        
        fig, axes = plt.subplots(len(self.classes), n_samples, 
                                figsize=(n_samples * 3, len(self.classes) * 2))
        fig.suptitle('Sample Images from Each Land Cover Class', fontsize=16)
        
        for i, class_name in enumerate(self.classes):
            class_samples = train_df[train_df['class_name'] == class_name].head(n_samples)
            
            for j in range(n_samples):
                ax = axes[i, j] if n_samples > 1 else axes[i]
                
                if j < len(class_samples):
                    img_path = class_samples.iloc[j]['full_path']
                    img = self.load_and_preprocess_image(img_path, normalize=False)
                    
                    if img is not None:
                        ax.imshow(img.astype(np.uint8))
                        ax.set_title(f'{class_name}')
                    else:
                        ax.text(0.5, 0.5, 'Failed to load', ha='center', va='center')
                        ax.set_title(f'{class_name} (Error)')
                else:
                    ax.text(0.5, 0.5, 'No sample', ha='center', va='center')
                    ax.set_title(f'{class_name} (No data)')
                
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.processed_data_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data pipeline.
        
        Returns:
            Dictionary of cleaned DataFrames
        """
        print("Starting EuroSAT Data Pipeline")
        
        # Step 1: Validate dataset structure
        self.validate_dataset_structure()
        
        # Step 2: Load label mapping
        self.load_label_map()
        
        # Step 3: Clean CSV files
        cleaned_dfs = self.clean_csv_files()
        
        # Step 4: Generate statistics
        if cleaned_dfs:
            self.generate_data_statistics(cleaned_dfs)
            self.create_sample_visualization(cleaned_dfs)
        
        print("\\n Data pipeline completed successfully")
        print(f" Processed data saved to: {self.processed_data_dir}")
        
        return cleaned_dfs

# Utility functions for external use
def load_processed_data(processed_dir: str = "data/processed") -> Dict[str, pd.DataFrame]:
    """
    Load processed CSV files.
    
    Args:
        processed_dir: Path to processed data directory
        
    Returns:
        Dictionary containing loaded DataFrames
    """
    processed_path = Path(processed_dir)
    dfs = {}
    
    for csv_file in ["train.csv", "test.csv", "validation.csv"]:
        csv_path = processed_path / csv_file
        if csv_path.exists():
            dfs[csv_file.replace('.csv', '')] = pd.read_csv(csv_path)
    
    return dfs

def get_image_batch(df: pd.DataFrame, batch_size: int = 32, 
                   target_size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a batch of images and their labels.
    
    Args:
        df: DataFrame containing image paths and labels
        batch_size: Number of images to load
        target_size: Target image size (height, width)
        
    Returns:
        Tuple of (images array, labels array)
    """
    pipeline = EuroSATDataPipeline()
    
    batch_df = df.sample(min(batch_size, len(df)))
    images = []
    labels = []
    
    for _, row in batch_df.iterrows():
        img = pipeline.load_and_preprocess_image(row['full_path'], target_size)
        if img is not None:
            images.append(img)
            labels.append(row['label'])
    
    return np.array(images), np.array(labels)

