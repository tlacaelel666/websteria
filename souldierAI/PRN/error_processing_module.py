try:
    with open('example.txt', 'r') as file:
        file_content = file.read()
except FileNotFoundError as e:
    print(f"Error: {e}. Please check if the file exists and the path is correct.")
    file_content = None
except PermissionError as e:
    print(f"Error: {e}. Ensure you have the necessary permissions to access the file.")
    file_content = None
except IsADirectoryError as e:
    print(f"Error: {e}. The specified path is a directory, not a file.")
    file_content = None
except IOError as e:
    print(f"Error: {e}. An I/O error occurred while accessing the file.")
    file_content = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    file_content = None

if file_content:
    print("File read successfully.")
