from datasets import load_dataset
import pandas as pd
from text_preprocessing import preprocess_text

def process_poem_sentiment():
    # Load the dataset
    ds = load_dataset("google-research-datasets/poem_sentiment")

    text_column = 'verse_text'

    # Convert each split to Parquet format and preprocess
    for split in ds.keys():
        df = pd.DataFrame(ds[split])
        
        # Apply preprocessing to the text column
        df[text_column] = df[text_column].apply(preprocess_text)
        
        # Save the preprocessed DataFrame back to Parquet format
        df.to_parquet(f'preprocessed_poem_sentiment_{split}.parquet')

    print("Preprocessing and saving complete.")

if __name__ == "__main__":
    process_poem_sentiment()