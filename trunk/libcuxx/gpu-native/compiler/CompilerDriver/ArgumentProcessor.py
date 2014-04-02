################################################################################
#
# \file   ArgumentProcessor.py
# \author Gregory Diamos <solusstultus@gmail.com>
# \date   Wednesday April 2, 2014
# \brief  The ArgumentProcessor tool, handles argument processor for CLANG.
#
################################################################################


class ArgumentProcessor:
	def __init__(self, arguments):
		self.arguments = arguments

		self.compilerArguments = []
		self.outputFile        = None
		self.inputFiles        = []

		self.parseArguments()


	def getCompilerArguments(self):
		return self.compilerArguments

	def getOutputFile(self):
		return self.outputFile

	def getInputFiles(self):
		return self.inputFiles
	

	def parseArguments(self):
		for argument in self.arguments:

			self.compilerArguments.append



		self.fillInOutputFile()



