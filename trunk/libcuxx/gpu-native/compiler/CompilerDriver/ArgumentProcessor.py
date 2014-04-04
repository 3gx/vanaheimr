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
		self.verbose           = False
		self.clean             = False

		self.strippedArguments = set()
		self.valueArguments    = set()

		self.parseArguments()


	def getCompilerArguments(self):
		return self.compilerArguments

	def getOutputFile(self):
		return self.outputFile

	def getInputFiles(self):
		return self.inputFiles
	
	def getVerbose(self):
		return self.verbose
	
	def getClean(self):
		return self.clean

	def parseArguments(self):
		self.outputFile = self.parseValue('-o')
		self.verbose    = self.parseExists('-v')
		
		self.strip('-o')
		
		self.argumentHasValue('-arct-migrate-report-output')
		self.argumentHasValue('-dependency-dot')
		self.argumentHasValue('-dependency-file')
		self.argumentHasValue('-F')
		self.argumentHasValue('-idirafter')
		self.argumentHasValue('-iframework')
		self.argumentHasValue('-I')
		self.argumentHasValue('-mllvm')
		self.argumentHasValue('-MQ')
		self.argumentHasValue('-MT')
		self.argumentHasValue('-serialize-diagnostics')
		self.argumentHasValue('-working-directory')
		
		self.fillInOutputFile()
		self.fillInArguments()

	def fillInOutputFile(self):
		if self.outputFile == None:
			self.outputFile = 'a.out'

	def strip(self, argument):
		self.strippedArguments.add(argument)

	def argumentHasValue(self, argument):
		self.valueArguments.add(argument)

	def parseValue(self, option):
		value = None
		
		found = False
		
		for argument in self.arguments:
			if found:
				value = argument
				break
				
			if argument == option:
				found = True
		
		return value

	def parseExists(self, option):
		return option in set(self.arguments)

	def fillInArguments(self):
		skip       = False
		isArgument = False
		arguments = []
	
		for argument in self.arguments:
			if skip:
				skip = False
				continue

			if argument in self.strippedArguments:
				skip = True
				continue
				
			if isArgument:
				isArgument = False
				arguments.append(argument)
				continue
			
			if argument in self.valueArguments:
				isArgument = True
			
				arguments.append(argument)
				continue
		
			if argument[0] == '-':
				arguments.append(argument)
				continue

			self.inputFiles.append(argument)
			

		#print 'parsed arguments: ' + str(arguments)
		#print 'parsed input files: ' + str(self.inputFiles)
	
		self.compilerArguments = arguments


