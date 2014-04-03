################################################################################
#
# \file   CLANG.py
# \author Gregory Diamos <solusstultus@gmail.com>
# \date   Tuesday June 25, 2013
# \brief  The CLANG, controls the C++ compiler driver.
#
################################################################################

# Compiler Modules
from FileTypes import *

# Standard Modules
from   which    import which
from   time     import time
import os
import subprocess

# The CLANG Compiler class
class CLANG:
	def __init__(self, driver):
		self.driver = driver

	def compile(self, files):
		
		outputs = []
		
		for filename in files:
			if self.canCompile(filename):
				outputs.append(self.lower(filename))
			else:
				outputs.append(filename)
		
		return outputs, self.isFinished()

	def isFinished(self):
		return isPTX(self.driver.outputFile)

	def getIntermediateFiles(self, filename):
		if self.canCompile(filename):
			if not self.isFinished():
				return [self.getOutputFilename(filename)]
		return []
	
	def canCompileFile(self, filename):
		if isCPP(filename):
			return True
				
		return False

	def canCompile(self, filenames):
		for filename in filenames:
			if not canCompileFile(filename):
				return False

		return True

	def getOutputFilename(self):
		if self.isFinished():
			return self.driver.outputFile
		else:
			return getBase(self.driver.outputFile) + '.ptx'

	def lower(self, filenames):
		assert self.canCompile(filenames)
		
		backend_path = which(self.getCLANGName())

		outputFilename = self.getOutputFilename()

		safeRemove(outputFilename)

		command = backend_path + " -S -target nvptx64 " + ' '.join(filenames) + " -o " + \
			outputFilename + " " + self.driver.getCompilerArguments()

		self.driver.getLogger().info('Running ' + self.getCLANGName() + ' with: "' + command + '"')
		
		start = time()

		process = subprocess.Popen(command, shell=True,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		(stdOutData, stdErrData) = process.communicate()
		
		self.driver.getLogger().info(' time: ' + str(time() - start) + ' seconds')

		if not os.path.isfile(outputFilename):
			raise SystemError(self.getCLANGName() + ' failed to generate an output file: \n' \
				+ stdOutData + stdErrData)

		return outputFilename

	def printHelp(self):
		
		backend_path = which(self.getCLANGName())
		
		command = backend_path + " --help"

		process = subprocess.Popen(command, shell=True,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		(stdOutData, stdErrData) = process.communicate()
	
		if stdOutData != None:	
			print stdOutData

		if stdErrData != None:
			print stdErrData
 
	def getCLANGName(self):
		return 'clang++'

	def getName(self):
		return "clang"


