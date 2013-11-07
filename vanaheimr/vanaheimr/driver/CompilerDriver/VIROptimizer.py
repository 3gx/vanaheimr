################################################################################
#
# \file   VIROptimizer.py
# \author Gregory Diamos <solusstultus@gmail.com>
# \date   Tuesday June 25, 2013
# \brief  The VIROptimizer, controls the VIR Optimizer tool.
#
################################################################################

from FileTypes import *

from   which    import which
from   time     import time
import os
import subprocess

# The VIROptimizer class
class VIROptimizer:
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
		return isByteCode(self.driver.outputFile)

	def getIntermediateFiles(self, filename):
		if self.canCompile(filename):
			if not self.isFinished():
				return [self.getOutputFilename(filename)]
		return []

	def canHandleAtLeastOne(self, files):
		for filename in files:
			if self.canCompile(filename):
				return True
										
		return False
	
	def canCompile(self, filename):
		if isLLVM(filename):
			return True
			
		return False

	def getOutputFilename(self, filename):
		if isByteCode(self.driver.outputFile):
			return self.driver.outputFile
		else:
			return getBase(filename) + '.bc'

	def lower(self, filename):
		assert self.canCompile(filename)

		backend_path = which('vir-optimizer')

		outputFilename = self.getOutputFilename(filename)

		safeRemove(outputFilename)

		command = backend_path + " -i " + filename + " -o " + \
			outputFilename + " " + \
			self.interpretOptimizations(self.driver.optimizationLevel)

		if self.driver.verbose:
			command += " -v"
		
		if self.driver.verbose:
			print 'Running vir-optimizer with: "' + command + '"'

		start = time()

		process = subprocess.Popen(command, shell=True,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		(stdOutData, stdErrData) = process.communicate()

		if self.driver.verbose:
			print ' time: ' + str(time() - start) + ' seconds' 

		if not os.path.isfile(outputFilename):
			raise SystemError('vir-optimizer failed to generate an ' + \
				'output file: \n' \
				+ stdOutData + stdErrData)

		return outputFilename

	def interpretOptimizations(self, level):
		return ""




