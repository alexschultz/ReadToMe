import os.path
from os import path
from shutil import rmtree, copyfile, make_archive
from distutils.dir_util import copy_tree
import zipfile


def main():
    print('packaging up ReadToMe lambda for deployment...')
    package_dir = 'lambda-package'
    zip_file = 'package.zip'
    if path.exists(package_dir):
        rmtree(package_dir)
    if path.exists(zip_file):
        os.remove(zip_file)

    os.mkdir(package_dir)
    copyfile('readToMe.py', os.path.join(package_dir, 'readToMe.py'))
    copyfile('imageProcessing.py', os.path.join(package_dir, 'imageProcessing.py'))
    copyfile('speak.py', os.path.join(package_dir, 'speak.py'))
    from_directory = os.path.join('local', 'lib', 'python2.7', 'site-packages')
    copy_tree(from_directory, package_dir)
    os.mkdir(os.path.join(package_dir, 'staticfiles'))
    copy_tree('staticfiles', os.path.join(package_dir, 'staticfiles'))

    make_archive('package', 'zip', package_dir)

    print('packaging complete')


if __name__ == "__main__":
    main()
