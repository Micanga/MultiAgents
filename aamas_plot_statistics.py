import subprocess

def is_constant(array):
	for i in range(0,len(array)-2):
		if array[i+1] - array[i] != 0:
			return False
	return True

def calcConfInt(p):
	f = open("tmp.R","w")
	f.write("#!/usr/bin/Rscript\n")

	listStr = ""

	for n in p:
		listStr = listStr + str(n) + ","

	f.write("print(t.test(c("+listStr[:-1]+"),conf.level=0.90))")

	f.close()

	os.system("chmod +x ./tmp.R")
	output = subprocess.check_output("./tmp.R",stderr=subprocess.STDOUT,shell=True)
	output = output.split()
	#print 'end of function', float(output[-7])
	return float(output[-7])