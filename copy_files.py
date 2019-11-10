import errno
import shutil
import os
import pickle

root_dir = "outputs"
files_type = "UCT"
#
#root_dir = "po_outputs"
#files_type = "POMCP"

def copy(src, dest):
    try:
        shutil.move(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)

count = 1 #UCT
#count = 29100  #POMCP
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
                itemsCounts = str(systemDetails['itemsCounts'])
		round = str(systemDetails['round_count'])

                if files_type =='POMCP':
                    radius = str(int(systemDetails['mainAgentRadius']))
			
                    if systemDetails['round_count']==1 or round == 'None' :
                        dest = 'tc/POMCP/p_s' + simWidth + '_a' + agentsCounts + '_i'+ itemsCounts + "_r" + radius + "/" + str(
                            count)
                        copy(root, dest )
                    else:
                        round = str(systemDetails['round_count'])
                        dest = 'tc/POMCP_mr/p_s' + simWidth + '_a' + agentsCounts + '_i'+ itemsCounts +"_r" + radius + "_ro" + round +  "/" + str(count)
                        copy(root, dest)
                else:
                    if systemDetails['round_count'] == 1 or round == 'None':
                        dest = 'tc/UCT/m_s' + simWidth + '_a' + agentsCounts + '_i'+ itemsCounts +"/" + str(count)
                        copy(root,dest)
                    else:
                        round = str(systemDetails['round_count'])
                        dest = 'categorised/UCT_mr/m_s' + simWidth + '_a' + agentsCounts + '_i'+ itemsCounts +"_ro" + round + "/" + str(count)
                        copy(root, dest)
                count +=1
                print dest ,'done'
                print '---------------------------------'
        else:
            f_count +=1
            print "No pickel:" + root

print'error count '+ str(f_count)
print count
