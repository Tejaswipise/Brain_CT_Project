from pathlib import Path

# Define project root folder (where config.py lives)
ROOT_DIR = Path(__file__).resolve().parent

# Define data folder (ROOT_DIR / 'data')
DATA_DIR = ROOT_DIR / 'data'

# Define results folder (ROOT_DIR / 'results')
RESULTS_DIR = ROOT_DIR / 'results'

# Now define the file paths using DATA_DIR
VALIDATED_METADATA = DATA_DIR / 'validated_metadata_clean.csv'
FEATURES_SLICES    = DATA_DIR / 'features_slices.csv'
FEATURES_PATIENTS  = DATA_DIR / 'features_patients.csv'
SCANNER_OFFSET_VECTOR = DATA_DIR / 'scanner_offset_vector.csv'