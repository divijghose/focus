import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        # ignore "__pycache__" directories
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        for f in files:
            # ignore .DS_Store files, *.pyc files, and directories
            if f == '.DS_Store':
                continue
            if f.endswith('.pyc'):
                continue
            if f == '__init__.py':
                continue
            print('{}{}'.format(subindent, f))

if __name__ == "__main__":
    list_files(os.getcwd())