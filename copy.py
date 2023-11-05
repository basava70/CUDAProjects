import os

directories = [i for i in os.listdir() if "." not in i]
for dir_ in directories:
    os.system("cp CPPLINT.cfg {}".format(dir_))
