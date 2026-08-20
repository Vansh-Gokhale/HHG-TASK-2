import os
import sys
import unittest.mock as mock

# Set up environment variables to avoid tf/keras issues
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Add the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Define mock dataset
def mock_load_dataset(*args, **kwargs):
    def mock_stream():
        for i in range(15):
            yield {
                "target_lang": "hi",
                "source_lang": "en",
                "query_id": i,
                "passages": {
                    "Translated_passages": [f"यह एक उदाहरण हिंदी वाक्य है {i}।"] * 3,
                    "is_selected": [1, 0, 0]
                }
            }
        for i in range(15):
            yield {
                "target_lang": "ta",
                "source_lang": "en",
                "query_id": i + 100,
                "passages": {
                    "Translated_passages": [f"இது ஒரு உதாரணம் தமிழ் வாக்கியம் {i}."] * 3,
                    "is_selected": [1, 0, 0]
                }
            }
    return mock_stream()

if __name__ == "__main__":
    import datasets
    # Patch load_dataset in datasets module globally so the pipeline picks it up
    datasets.load_dataset = mock_load_dataset
    datasets.get_dataset_config_names = lambda *args, **kwargs: ["default"]

    from ingestion.main import main
    print("Running ingestion pipeline for 10 samples of Hindi ('hi') and Tamil ('ta')...")
    # We pass the arguments to main
    test_args = [
        "--languages", "hi,ta",
        "--limit", "10",
        "--fresh", 
        "--output-dir", "./test_lancedb_data"
    ]
    main(test_args)
    
    print("\nDone! To check your vector db, you can use lancedb in python:")
    print('''
import lancedb
db = lancedb.connect("./test_lancedb_data")
print("Tables:", db.table_names())
table_hi = db.open_table("hi")
print("Hindi Table head:\\n", table_hi.to_pandas().head())
    ''')
