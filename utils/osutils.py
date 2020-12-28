from __future__ import absolute_import 
import os
import os.path as osp
import errno
import yaml
import shutil 

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise




