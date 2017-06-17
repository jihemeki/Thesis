import os
import subprocess

while True:
    process = subprocess.Popen(["python","__init__.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    try:
        if stdout[0] == "1":
            os.system('python Luu.py')
    except IndexError:
	print "Exit"
	break


"""
process = subprocess.Popen(["python","__init__.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout,stderr = process.communicate()
print stdout[]
"""
