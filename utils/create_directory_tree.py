import os


def create_directory_tree(full_filename):
    full_filename_split = full_filename.split('\\')
    full_filename_reconstruction = ''
    for n in range(len(full_filename_split)):
        full_filename_reconstruction += full_filename_split[n] + '\\'
        if not os.path.isdir(full_filename_reconstruction):
            os.mkdir(full_filename_reconstruction)