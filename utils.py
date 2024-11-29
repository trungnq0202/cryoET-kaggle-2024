import os

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_multiple_dirs(dir_list):
    for dir in dir_list:
        create_dir(dir)