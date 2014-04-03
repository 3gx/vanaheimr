################################################################################
#
# \file   PTXLINK.py
# \author Gregory Diamos <solusstultus@gmail.com>
# \date   Tuesday June 25, 2013
# \brief  The PTXLINK tool, controls the PTX linker.
#
################################################################################

# Compiler Modules
from FileTypes import *

# Standard Modules
from   which    import which
from   time     import time
import os
import subprocess

# The PTXLINK Compiler class
class PTXLINK:
	def __init__(self, driver):
		self.driver = driver

	def compile(self, files):
		
		outputs = self.link(files)	
	
		return outputs, self.isFinished()

	def isFinished(self):
		return isPTX(self.driver.outputFile)

	def getIntermediateFiles(self, filename):
		if self.canCompile(filename):
			if not self.isFinished():
				return [self.getOutputFilename(filename)]
		return []

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

	def link(self, filenames):
		assert self.canCompile(filenames)
		
		outputFilename = self.getOutputFilename()

		safeRemove(outputFilename)

		self.driver.getLogger().info('Running PTXLINK...')

		start = time()

		for filename in filenames:
			self.append(outputFileName, filename)

		self.driver.getLogger().info(' time: ' + str(time() - start) + ' seconds')

		if not os.path.isfile(outputFilename):
			raise SystemError(getCLANGName() + ' failed to generate an output file: \n' \
				+ stdOutData + stdErrData)

		return outputFilename

	def append(self, output, input):
		outputFile = open(output, 'a')
		inputFile  = open(input, 'r')

		for line in inputFile:
			outputFile.write(line)

	def getName(self):
		return "ptxlink"

