import json
import os

def export_to_json(output_file='project_files.json'):
    files_to_export = [
        'fraud_train.py',
        'app.py',
        'streamlit_app.py',
        'requirements.txt',
        'README.md',
        'test_api.py',
        'fraud_detection.py'
    ]
    
    project_data = {}
    
    for filename in files_to_export:
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    project_data[filename] = f.read()
                print(f"Added {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        else:
            print(f"File not found: {filename}")
            
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=4)
        print(f"Successfully exported files to {output_file}")
    except Exception as e:
        print(f"Error writing JSON file: {e}")

if __name__ == "__main__":
    export_to_json()
