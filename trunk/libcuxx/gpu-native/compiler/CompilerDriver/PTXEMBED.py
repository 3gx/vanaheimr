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
		return []
	
	def canCompileFile(self, filename):
		if isPTX(filename):
			return True
				
		return False

	def canCompile(self, filenames):
		if len(filenames) > 1:
			return False

		for filename in filenames:
			if not self.canCompileFile(filename):
				return False

		return True

	def getOutputFilename(self):
		return self.driver.outputFile

	def getTemporaryFilename(self):
		return getBase(self.getOutputFilename()) + '.cpp'

	def embed(self, filenames):
		assert self.canCompile(filenames)
		
		outputFilename = self.getOutputFilename()

		safeRemove(outputFilename)

		self.driver.getLogger().info('Running PTXEMBED...')

		start = time()
		
		tempFilename = self.getTemporaryFilename()

		self.embedPTX(tempFilename, filenames[0])
		
		(stdOutData, stdErrData) = self.compileAndLink(outputFilename, tempFilename)
		safeRemove(tempFilename)
		
		self.driver.getLogger().info(' time: ' + str(time() - start) + ' seconds')

		if not os.path.isfile(outputFilename):
			raise SystemError('ptx embed failed to generate an output file!' +
				stdOutData + stdErrData)

		return outputFilename

	def embedPTX(self, output, input):
		outputFile = open(output, 'a')

		outputFile.write("\n")
		outputFile.write("static char ptx[] = {")
		
		counter   = 0
		threshold = 8
		
		with open(input, 'rb') as inputFile:
			byte = inputFile.read(1)
			
			while byte:
				outputFile.write(hex(ord(byte)) + ", ")
				
				counter += 1
				
				if counter >= threshold:
					counter = 0
					outputFile.write("\n")
		
				byte = inputFile.read(1)

		outputFile.write("0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};\n")

		outputFile.write("\n\n")
		outputFile.write("extern void setEmbeddedPTX(const char*);\n")
		
		outputFile.write("\n\n")
		outputFile.write("extern int gpuNativeMain(int argc, const char** argv);\n")

		outputFile.write("\n\n")
		outputFile.write("extern int main(int argc, const char** argv)\n")
		outputFile.write("{\n")
		outputFile.write("\tsetEmbeddedPTX(ptx);\n")
		outputFile.write("\treturn gpuNativeMain(argc, argv);\n")
		outputFile.write("}\n")
		
		outputFile.write("\n\n\n\n\n")

	def compileAndLink(self, output, input):
		
		embedderPath = which(self.getCxxCompiler())
		
		command = embedderPath + " -o " + output + ' ' + input
		command += ' ' + self.getRuntimeLibraryArguments()
		command += ' ' + ' '.join(self.driver.getCompilerArguments())

		self.driver.getLogger().info(' command is ' + command)
	
		process = subprocess.Popen(command, shell=True,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE)

		(stdOutData, stdErrData) = process.communicate()

		return stdOutData, stdErrData

	def getRuntimeLibraryArguments(self):
		return '-L' + os.path.join(self.driver.getScriptPath(), '..', 'lib') + ' -lgpunative'

	def getCxxCompiler(self):
		cxx = os.environ.get('CXX')
		
		if cxx == None:
			# TODO: search for more possibilities
			cxx = 'clang++'

		return cxx

	def getName(self):
		return "ptxembed"
		


