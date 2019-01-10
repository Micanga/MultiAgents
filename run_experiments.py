
import sys
import os

config_path = None

for dirs in os.listdir("inputs/2"):

    for files in os.walk(""
                         ""
                         "inputs/2/" + str(dirs)):
        config_path = str(files[0]) + '/'

        print config_path
        os.system('python po_run_world.py ' + config_path)
                      # + str(files[2][0])




