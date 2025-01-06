import os

def clear_directory(directory):
    for filename in os.listdir(directory):
        if filename == 'clear.py':
            continue
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    clear_directory(current_directory)