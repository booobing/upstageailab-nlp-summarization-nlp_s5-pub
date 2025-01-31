# Install required libraries
# !pip install pandas
# !pip install googletrans==3.1.0a0

import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Read input data
input_path = '/home/fine_data/train.csv'  # 원본 데이터 경로
data = pd.read_csv(input_path)

# Initialize the translator
translator = Translator()

# Function to perform back translation
def back_translate(text, source_lang='ko', target_lang='en'):
    try:
        # Translate to English
        translated = translator.translate(text, src=source_lang, dest=target_lang).text
        # Translate back to Korean
        back_translated = translator.translate(translated, src=target_lang, dest=source_lang).text
        return back_translated
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return text

# Apply back translation on the 'dialogue' column with progress bar
logging.info("Starting back translation process...")
tqdm.pandas(desc="Translating")
data['dialogue'] = data['dialogue'].progress_apply(lambda x: back_translate(x))
logging.info("Translation completed!")

# Save the updated data to a new CSV file
output_path = '/home/fine_data/back_translated_train.csv'
data.to_csv(output_path, index=False)

output_path