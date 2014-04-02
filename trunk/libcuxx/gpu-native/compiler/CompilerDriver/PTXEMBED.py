################################################################################
#
# \file   PTXEMBED.py
# \author Gregory Diamos <solusstultus@gmail.com>
# \date   Tuesday June 25, 2013
# \brief  The PTXEMBED tool, controls the PTX embeder.
#
################################################################################

# Compiler Modules
from FileTypes import *

# Standard Modules
from   which    import which
from   time     import time
import os
import subprocess

# The PTXEMBED Compiler class
class PTXEMBED:
	def __init__(self, driver):
		self.driver = driver

	def compile(self, files):
		
		outputs = self.embed(files)	
	
		return outputs, self.isFinished()

	def isFinished(self):
		return True 

	def getIntermediateFiles(self, filename):
		return [getBase(filename) + '-embedded.cpp']
	
	def canCompileFile(self, filename):
		if isPTX(filename):
			return True
				
		return False

	def canCompile(self, filenames):
		if(len(filenames) < 2):
			return False

		for filename in filenames:
			if not canCompileFile(filename):
				return False

		return True

	def getOutputFilename(self):
		if self.isFinished():
			return self.driver.outputFile
		else:
			return getBase(self.driver.outputFile) + '.ptx'

	def embed(self, filenames):
		assert self.canCompile(filenames)
		
		outputFilename = self.getOutputFilename()

		safeRemove(outputFilename)

		self.driver.getLogger().info('Running PTXEMBED...')

		start = time()
		
		self.embedPTX(outputFileName, filenames[0])
		
		self.driver.getLogger().info(' time: ' + str(time() - start) + ' seconds')

		if not os.path.isfile(outputFilename):
			raise SystemError('ptx embed failed to generate an output file!')

		return outputFilename

	def embedPTX(self, output, input):
		outputFile = open(output, 'a')

		outputFile.write("\n")
		outputFile.write("static char ptx[] = {")
		
		counter   = 0
		threshold = 8
		
		with open(input, 'rb') as inputFile:
			byte = inputFile.read(1)
			
			if byte:
				outputFile.write(hex(byte) + ", ")
				
				counter += 1
				
				if counter >= threshold:
					counter = 0
					outputFile.write("\n")
		
		outputFile.write("0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};\n\n")

		outputFile.write("extern const char* getEmbeddedPTX()\n")
		outputFile.write("{\n")
		outputFile.write("\treturn ptx;\n")
		outputFile.write("}\n")
		outputFile.write("\n\n\n\n\n")
		


