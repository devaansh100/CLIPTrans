import os
import sys

def main(list_file_path, source_folder, destination_folder):
    with open(list_file_path, 'r') as file_list:
        for line in file_list:
            # Strip newline character and any leading/trailing spaces from the line
            file_name = line.strip()

            # Check if the file exists before moving
            if os.path.isfile(file_name):
                # Use os.path.basename to get the filename without the path
                file_name_only = os.path.basename(file_name)

                # Move the file to the destination folder
                source_path = os.path.join(source_folder, file_name_only)
                destination_path = os.path.join(destination_folder, file_name_only)
                os.rename(source_path, destination_path)
            else:
                print(f"File not found: {file_name}")

if __name__ == '__main__':
    file_list_path = sys.argv[1]
    source_folder = sys.argv[2]
    destination_folder = sys.argv[3]
    main(file_list_path, source_folder, destination_folder)
