################################################################################
#
# \file   FileTypes.py
# \author Gregory Diamos <solusstultus@gmail.com>
# \date   Tuesday June 25, 2013
# \brief  Functions for manipulating files.
#
################################################################################

import os

# Global Functions
def getBase(string):
	while True:
		split = os.path.splitext(string)
		string = split[0]
		if split[1] == '':
			break
			
	return string

def getSpecificExt(string):
	extension = ''
	
	while True:
		split = os.path.splitext(string)
		
		string = split[0]
		ext    = split[1]
	
		if ext.find('.') != 0:
			break

		extension = ext + extension
		
	return extension


def getExt(string):
	extension = os.path.splitext(string)[1]

	return extension

def safeRemove(string):
	if(os.path.isfile(string)):
		os.remove(string)

def getTreePath(base):
	head, tail = os.path.split(os.path.realpath(os.getcwd()))

	while len(tail) > 0:
		if tail == base:
			return head
		
		head, tail = os.path.split(head)
		
	assert False, "Could not find tree root (e.g. //" + base + "/...) ."

def isPTX(string):
	return getExt(string) == '.ptx'

def isVIR(string):
	return getExt(string) == '.vir'

def isLLVM(string):
	return getExt(string) == '.llvm'

def isByteCode(string):
	return getExt(string) == '.bc'

def isCPP(string):
	return getExt(string) == '.cpp'

def isC(string):
	return getExt(string) == '.c'

def isTarFile(string):
	return getSpecificExt(string) == '.tar.gz'

