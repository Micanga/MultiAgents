import errno
import shutil

import os

import pickle
root_dir = "outputs"
files_type = "UCT"
#
# root_dir = "p_tmp_output"
# files_type = "POMCP"

def copy(src, dest):
    try:
        shutil.move(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)

count = 6500
f_count = 0
for root, dirs, files in os.walk(root_dir):
        if 'pickleResults.txt' in files:
            print root
            with open(os.path.join(root,'pickleResults.txt'),"r") as pickleFile:
                dataList = pickle.load(pickleFile)

                # Simulator Information
                systemDetails = dataList[0]
                estimationDictionary = {}
                simWidth = str(systemDetails['simWidth'])
                simHeight = systemDetails['simHeight']
                agentsCounts = str(systemDetails['agentsCounts'])
                itemsCounts = systemDetails['itemsCounts']
                if files_type =='POMCP':
                    radius = str(int(systemDetails['mainAgentRadius']))

                    if systemDetails['round_count']==1:
                        dest = 'categorised/POMCP/p_s' + simWidth + '_a' + agentsCounts + "_r" + radius + "/" + str(
                            count)
                        copy(root, dest )
                    else:
                        round = str(systemDetails['round_count'])
                        dest = 'categorised/POMCP_mr/p_s' + simWidth + '_a' + agentsCounts + "_r" + radius + "_ro" + round +  "/" + str(count)
                        copy(root, dest)
                else:
                    if systemDetails['round_count']==1:
                        dest = 'tc/UCT/m_s' + simWidth + '_a' + agentsCounts + "/" + str(count)
                        copy(root,dest)
                    else:
                        round = str(systemDetails['round_count'])
                        dest = 'categorised/UCT_mr/m_s' + simWidth + '_a' + agentsCounts + "_ro" + round + "/" + str(count)
                        copy(root, dest)
                count +=1
                print dest ,'done'
                print '---------------------------------'
        else:
            f_count +=1
            print "No pickel:" + root

print'error count '+ str(f_count)
print count