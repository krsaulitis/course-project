import os


def list_files(directory):
    """ List all files in a directory (excluding subdirectories) """
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]


def find_unique_filenames(dir1, dir2):
    """ Find filenames that are not common in two directories """
    files_dir1 = set(list_files(dir1))
    files_dir2 = set(list_files(dir2))

    unique_filenames_dir1 = files_dir1.difference(files_dir2)
    unique_filenames_dir2 = files_dir2.difference(files_dir1)

    return unique_filenames_dir1, unique_filenames_dir2


# Replace these paths with the paths of your directories
directory1 = '../inference/mary_tts/audios/'
directory2 = '../inference/glow_tts/audios/'

unique_filenames_dir1, unique_filenames_dir2 = find_unique_filenames(directory1, directory2)

if unique_filenames_dir1 or unique_filenames_dir2:
    print("Unique filenames in the first directory:")
    for filename in unique_filenames_dir1:
        print(filename)
    print("\nUnique filenames in the second directory:")
    for filename in unique_filenames_dir2:
        print(filename)
else:
    print("All filenames in both directories are the same.")
