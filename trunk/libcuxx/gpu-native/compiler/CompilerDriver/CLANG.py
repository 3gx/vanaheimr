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
		
		return [self.lower(files)], self.isFinished()

	def isFinished(self):
		return isPTX(self.driver.outputFile)

	def getIntermediateFiles(self, filenames):
		if self.canCompile(filenames):
			return [self.getOutputFilename()]
		return []
	
	def canCompileFile(self, filename):
		if isCPP(filename):
			return True
				
		return False

	def canCompile(self, filenames):
		for filename in filenames:
			if not self.canCompileFile(filename):
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

		command = backend_path + " -D __NV_CLANG__ -S -target nvptx64 " + ' '.join(filenames) + " -o " + \
			outputFilename + " " + ' '.join(self.driver.getCompilerArguments())

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
		llvmPath = os.environ.get('LLVM_INSTALL_PATH')
		
		if llvmPath == None:
			return 'clang++'

		return os.path.join(llvmPath, 'bin', 'clang++')

	def getName(self):
		return "clang"


