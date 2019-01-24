
import sys
import os

config_path = None

for dirs in os.listdir("inputs/FO"):

    for files in os.walk(""
                         ""
                         "inputs/FO/" + str(dirs)):
        config_path = str(files[0]) + '/'

        print config_path
        os.system('python run_world.py ' + config_path)
                      # + str(files[2][0])




