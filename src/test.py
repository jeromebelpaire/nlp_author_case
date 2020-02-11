import os
ROOT_DIR = os.getcwd()

TEST_DIR= "%s/src/test"%ROOT_DIR
print(type(TEST_DIR))
print(os.listdir(TEST_DIR))