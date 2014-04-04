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
		return '' != self.getCompilerMode()
 
	def getIntermediateFiles(self, filenames):
		if self.canCompile(filenames):
			if not self.isFinished():
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

		command = backend_path + " -D __NV_CLANG__ -target nvptx64 " + ' '.join(filenames) + " -o " + \
			outputFilename

		command += " " + self.getCompilerModeFlag()
		command += " " + ' '.join(self.getCompilerArguments())
		command += " " + ' '.join(self.getSystemIncludeFlags())

		self.driver.getLogger().info('Running ' + self.getCLANGName() + ' with: "' + command + '"')
		
		start = time()

		process = subprocess.Popen(command, shell=True,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		(stdOutData, stdErrData) = process.communicate()
		
		self.driver.getLogger().info(' time: ' + str(time() - start) + ' seconds')

		if not os.path.isfile(outputFilename):
			errorString = self.getCLANGName() + ' failed to generate an output file: \n'
			errorString += ' command: ' + command + '\n'
			errorString += ' stdout: ' + stdOutData + '\n'
			errorString += ' stderr: ' + stdErrData + '\n'

			raise SystemError(errorString)

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

	def getSystemIncludePaths(self):
		binPath = self.driver.getScriptPath()
		
		includePath = os.path.join(binPath, '..', 'include')

		return [os.path.join(includePath, 'libcxx', 'include'), 
			os.path.join(includePath, 'libc'),
			os.path.join(includePath, 'libcuxx')]

	def getSystemIncludeFlags(self):
		return ['-I' + path for path in self.getSystemIncludePaths()]

	def isCompilerMode(self, mode):
		
		modes = set(['-c', '-E', '-S'])
		
		return mode in modes

	def getCompilerMode(self):
		mode = ""

		for argument in self.driver.getCompilerArguments():
			if self.isCompilerMode(argument):
				mode = argument

		return mode

	def getCompilerModeFlag(self):
		mapping = { '' : '-S',
			'-S' : '-S',
			'-E' : '-E',
			'-c' : '-S'}

		return mapping[self.getCompilerMode()]

	def getCompilerArguments(self):
		arguments = []

		# strip modes
		for argument in self.driver.getCompilerArguments():
			if self.isCompilerMode(argument):
				continue

			arguments.append(argument)

		return arguments


	def getName(self):
		return "clang"


