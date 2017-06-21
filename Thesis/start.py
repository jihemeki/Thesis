import os
import subprocess

while True:
    process = subprocess.Popen(["python","__init__.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    try:
        if stdout[0] == "1":
            os.system('python CameraSolve.py')
        elif stdout[0] == "2":
	    os.system('python CameraHint.py')
        else:
	    print "Exit"
	    break
    except IndexError:
        print "Exit"
        break

#This is a tesxt
