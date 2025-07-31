import re
from os.path import dirname, abspath


def get_root_path():
    return dirname(abspath(__file__))

def create_dir_if_not_exists(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def atoi(text):
    return int(text) if text.isdigit() else text
        
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]