import os
import sys


def prepare_file_system():
    # Ensure the 'src' directory is in the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.'))
    src_path = os.path.join(project_root, 'src')
    data_path = os.path.join(src_path, 'data')

    os.makedirs(data_path, exist_ok=True)

    if src_path not in sys.path:
        sys.path.append(src_path)

prepare_file_system()