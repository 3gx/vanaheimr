################################################################################
#
# \file   CLANG.py
# \author Gregory Diamos <solusstultus@gmail.com>
# \date   Tuesday June 25, 2013
# \brief  The CLANG, controls the C++ compiler driver.
#
################################################################################

from FileTypes import *

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
		
		return outputs, False

	def getIntermediateFiles(self, filename):
		if self.canCompile(filename):
			return [self.getOutputFilename(filename)]
		return []

	def canHandleAtLeastOne(self, files):
		for filename in files:
			if self.canCompile(filename):
				return True
										
		return False
	
	def canCompile(self, filename):
		if isCPP(filename):
			return True
				
		return False

	def getOutputFilename(self, filename):
		return getBase(filename) + '.llvm'

	def lower(self, filename):
		assert self.canCompile(filename)

		backend_path = which('clang++')

		outputFilename = self.getOutputFilename(filename)

		safeRemove(outputFilename)

		command = backend_path + " -emit-llvm " + filename + " -o " + \
			outputFilename + " " + \
			self.interpretOptimizations(self.driver.optimizationLevel)

		if self.driver.verbose:
			command += " -v"
		
		if self.driver.verbose:
			print 'Running clang++ with: "' + command + '"'

		start = time()

		process = subprocess.Popen(command, shell=True,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		(stdOutData, stdErrData) = process.communicate()

		if self.driver.verbose:
			print ' time: ' + str(time() - start) + ' seconds' 

		if not os.path.isfile(outputFilename):
			raise SystemError('clang++ failed to generate an output file: \n' \
				+ stdOutData + stdErrData)

		return outputFilename

	def interpretOptimizations(self, level):
		return ""



