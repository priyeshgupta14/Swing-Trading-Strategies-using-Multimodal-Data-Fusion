import os

def remove_merged_and_save(source_folder, destination_folder):
    """
    Removes "_merged" from file names in the source folder and saves the renamed files in the destination folder.

    Args:
        source_folder (str): The path to the source folder containing the files.
        destination_folder (str): The path to the destination folder where the renamed files will be saved.
    """

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        # Check if the file name contains "_merged"
        if "_merged" in filename:
            # Remove "_merged" from the file name
            new_filename = filename.replace("_merged", "")

            # Get the full paths for the old and new file names
            old_path = os.path.join(source_folder, filename)
            new_path = os.path.join(destination_folder, new_filename)

            # Rename the file
            os.rename(old_path, new_path)

# Example usage:
source_folder = "market_sentiments"
destination_folder = "news_sentiments"
remove_merged_and_save(source_folder, destination_folder)