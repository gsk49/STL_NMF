import subprocess
import logging
import os

# Set up logging
logging.basicConfig(filename="output_log.txt", level=logging.INFO, format='%(asctime)s - %(message)s')

# Directory containing the files
directory_path = "spaDecon_env_x86/lib/python3.9/site-packages"

# Loop through all files in the directory
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        
        try:
            # Run the 'file' command and capture output
            output = subprocess.check_output(["file", file_path]).decode()
            logging.info(f"Checked file: {file_path}\n{output}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error while processing file {file_path}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error with file {file_path}: {e}")
