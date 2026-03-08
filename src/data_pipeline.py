"""
Data pipeline - load and process 20 Newsgroups dataset
"""
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from typing import List, Tuple, Dict
import os


class DataPipeline:
    """Handle data loading and preprocessing"""
    
    def __init__(self, cache_dir: str = "./data"):
        """
        Initialize data pipeline
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Pipeline] Cache directory: {cache_dir}")
    
    def load_20newsgroups(self, subset: str = "train", categories: List[str] = None, 
                          remove_headers: bool = True) -> Tuple[List[str], List[str]]:
        """
        Load 20 Newsgroups dataset
        
        Args:
            subset: 'train', 'test', or 'all'
            categories: Specific categories to load (None = all)
            remove_headers: Whether to remove headers/quotes
            
        Returns:
            Tuple of (texts, categories)
        """
        print(f"[Pipeline] Loading 20 Newsgroups ({subset})...")
        
        newsgroups = fetch_20newsgroups(
            subset=subset,
            categories=categories,
            remove=('headers', 'footers', 'quotes') if remove_headers else (),
            download_if_missing=True,
            data_home=self.cache_dir
        )
        
        texts = newsgroups.data
        categories = newsgroups.target_names
        
        print(f"[Pipeline] Loaded {len(texts)} documents from {len(categories)} categories")
        return texts, categories
    
    def preprocess_texts(self, texts: List[str], max_length: int = 512, min_length: int = 10) -> Tuple[List[str], List[int]]:
        """
        Preprocess texts - clean and filter
        
        Args:
            texts: List of raw texts
            max_length: Maximum text length in words
            min_length: Minimum text length in words
            
        Returns:
            Tuple of (cleaned_texts, valid_indices)
        """
        print(f"[Pipeline] Preprocessing {len(texts)} texts...")
        
        cleaned_texts = []
        valid_indices = []
        
        for idx, text in enumerate(texts):
            # Clean text
            text = text.strip()
            words = text.split()
            
            # Filter by length
            if min_length <= len(words) <= max_length:
                cleaned_texts.append(text)
                valid_indices.append(idx)
        
        print(f"[Pipeline] After filtering: {len(cleaned_texts)} texts retained")
        print(f"[Pipeline] Removed {len(texts) - len(cleaned_texts)} texts (too short/long)")
        
        return cleaned_texts, valid_indices
    
    def split_data(self, texts: List[str], train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
        """
        Split data into train and test
        
        Args:
            texts: List of texts
            train_ratio: Ratio of training data
            
        Returns:
            Tuple of (train_texts, test_texts)
        """
        split_idx = int(len(texts) * train_ratio)
        train_texts = texts[:split_idx]
        test_texts = texts[split_idx:]
        
        print(f"[Pipeline] Split data: {len(train_texts)} train, {len(test_texts)} test")
        return train_texts, test_texts
    
    @staticmethod
    def sample_texts(texts: List[str], n_samples: int = 1000) -> List[str]:
        """
        Sample random texts
        
        Args:
            texts: List of texts
            n_samples: Number of samples
            
        Returns:
            Sampled texts
        """
        if len(texts) <= n_samples:
            return texts
        
        indices = np.random.choice(len(texts), n_samples, replace=False)
        sampled = [texts[i] for i in indices]
        print(f"[Pipeline] Sampled {n_samples} texts from {len(texts)} total")
        return sampled
