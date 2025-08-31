from datasets import Dataset
import pyarrow as pa

def print_first_audio_phonemes(file_path):
    """Simply print the phonemes for the first audio file"""
    
    try:
        dataset = Dataset.from_file(file_path)
        
        if 'text' in dataset.features:
            print("Audio File:")
            print(dataset[0]['audio_path'])
            first_text = dataset[0]['text']
            print("Text for first audio file:")
            print(first_text)

            # Additional info if it's a list/array
            if isinstance(first_text, (list, tuple)):
                print(f"Number of text elements: {len(first_text)}")
            elif hasattr(first_text, 'shape'):
                print(f"Text array shape: {first_text.shape}")

        else:
            print("No 'text' column found. Available columns:")
            print(list(dataset.features.keys()))
            
    except Exception as e:
        print(f"Error: {e}")

# Usage
if __name__ == "__main__":
    print_first_audio_phonemes("M:/git/github/celophi/F5-TTS/data/Bilingual_EN_JP_custom/raw.arrow")