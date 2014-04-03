################################################################################
#
# \file   CompilerDriver.py
# \author Gregory Diamos <gregory.diamos@gmail.com>
# \date   Tuesday April 1, 2014
# \brief  A class for the clang-ptx compiler driver.  It controls other modules.
#
################################################################################

from CLANG             import CLANG
from PTXLINK           import PTXLINK
from PTXEMBED          import PTXEMBED
from ArgumentProcessor import ArgumentProcessor

import os
import logging
import sys

# The Compiler Driver class
class CompilerDriver:
	def __init__(self):
		self.knobs = []
		self.arguments = sys.argv[1:]

		self.components        = []
		self.intermediateFiles = set([])

		self.compilerArguments = "" 
		self.outputFile        = ""
		self.inputFiles        = []
		self.verbose           = False
		self.clean             = False
		

	def getCompilerArguments(self):
		return self.compilerArguments
	
	def registerComponents(self):
		self.registerComponent(CLANG(self))
		self.registerComponent(PTXLINK(self))
		self.registerComponent(PTXEMBED(self))

	def run(self):
		self.registerComponents()
		
		self.validateInputs()
		
		if self.clean:
			self.cleanIntermediateFiles()
			return
		
		self.compile()

	def cleanIntermediateFiles(self):
		self.logger.info("Removing intermediate files")
		
		for filename in self.intermediateFiles:
			self.logger.info(" " + filename)
			safeRemove(filename)
	
	def compile(self):
		self.compileFile(self.outputFile, self.inputFiles)
		
		self.logger.info("Compilation Succeeded")	
		
	def compileFile(self, outputFile, inputFiles):
	
		
		self.logger.info("Compiling " + str(inputFiles) + " to " + outputFile)
	
		oldOutputFile    = self.outputFile
		oldIntermediates = self.intermediateFiles

		self.outputFile = outputFile
	
		self.computeIntermediateFiles(inputFiles)

		currentFiles = inputFiles
		
		finished = False
		
		while not finished:
			compiler = self.getCompilerThatCanHandle(currentFiles)
			
			if compiler == None:
				raise SystemError("No component can handle the "
					"intermediate files " + str(currentFiles))
						
			currentFiles, finished = compiler.compile(currentFiles)
	
		if not self.keep:
			self.cleanIntermediateFiles()
	
		self.outputFile        = oldOutputFile
		self.intermediateFiles = oldIntermediates
	
	def registerComponent(self, component):
		self.components.append(component)
	
	def computeIntermediateFiles(self, inputFiles):

		self.intermediateFiles = set(inputFiles)

		changed = True
		length  = 0
		
		while changed:
			for compiler in self.components:
				intermediates = []
				
				for intermediate in self.intermediateFiles:
					intermediates += compiler.getIntermediateFiles(intermediate)
				
				self.intermediateFiles.update(intermediates)
	
			changed = (length != len(self.intermediateFiles))
			length = len(self.intermediateFiles)
		
		for file in inputFiles:
			self.intermediateFiles.remove(file)
		
	def getCompilerThatCanHandle(self, files):
		for component in self.components:
			if component.canCompile(files):
				return component
	
		return None

	def validateInputs(self):
		processor = ArgumentProcessor(self.arguments)
		
		self.compilerArguments = processor.getCompilerArguments()
		self.outputFile        = processor.getOutputFile()
		self.inputFiles        = processor.getInputFiles()
		self.verbose           = processor.getVerbose()
		self.clean             = processor.getClean()

		self.loadLogger()

		if len(self.inputFiles) == 0:
			raise ValueError("No input file specified.")

	def loadLogger(self):
		self.logger = logging.getLogger('ClangPTXCompiler')
		logging.basicConfig()
		
		if self.verbose:
			self.logger.setLevel(logging.DEBUG)
		else:
			self.logger.setLevel(logging.ERROR)

		self.logger.info("Loaded knobs: " + str(self.knobs))

	def printHelp(self):
		clang = self.getComponent("clang")

		clang.printHelp()

	def getLogger(self):
		return self.logger

	def getComponent(self, name):
		for component in self.components:
			if component.getName() == name:
				return component

		return None

