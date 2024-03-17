import os
from typing import *

def write_log(ss:Any)->None:
    """
        logs inputs in log file
        path is the same as the root where the main.py code is executed
        input: string type
    """
    with open("log.txt", "a") as fp:
        fp.write(str(ss)+"\n")
    fp.close()

def write_captured_id(ss:Any)->None:
    """
        writes id of captures in capture directory where main.py code is executed
    """
    if not os.path.isdir("captures"):
        pass
    else:
        os.mkdir("captures")

    with open("captures/patients.txt", "a") as fp:
        fp.write(str(ss)+"\n")
    fp.close()

def write_last_id(id:int):
    """ 
        writes last id of captures
    """
    f = open('last_id','w')
    f.write(str(id))
    f.close()

def read_last_id():
    """ 
        reads last id that has been captured
    """
    id = 0
    if os.path.exists('last_id'):
        f = open('last_id','r')
        id = f.readline()
        if id.isdigit():
            id = int(id)+1
        else:
            id = 0
            write_log("id file is empty. id reset to ~0~ !!!")
    return id
