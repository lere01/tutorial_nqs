import os
import sys



# Ensure the 'src' directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.'))

paths = ("observables")

for p in paths:
    src_path = os.path.join(project_root, p)

    if src_path not in sys.path:
        sys.path.append(src_path)