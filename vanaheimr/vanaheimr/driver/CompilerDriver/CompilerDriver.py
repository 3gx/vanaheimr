################################################################################
#
# \file   CompilerDriver.py
# \author Gregory Diamos <solusstultus@gmail.com>
# \date   Tuesday June 25, 2013
# \brief  A class for the vanaheimr compiler driver.  It controls other modules.
#
################################################################################

from CLANG        import CLANG
from VIROptimizer import VIROptimizer

from FileTypes import *

import os
import logging

# The Compiler Driver class
class CompilerDriver:
	def __init__(self, arguments):
		self.inputFile         = arguments['input_file']
		self.outputFile        = arguments['output_file']
		self.optimizationLevel = int(arguments['optimization_level'])
		self.verbose           = arguments['verbose']
		self.keep              = arguments['keep']
		self.clean             = arguments['clean']
		self.machineModel      = arguments['machine_model']
		self.knobs             = interpretKnobs(arguments['knob'])
		self.onlyAssemble      = arguments['assembly']

		self.components        = []
		self.intermediateFiles = set([])

		self.loadLogger()

		self.validateInputs()
				
		self.registerComponents()
	
	def registerComponents(self):
				
		self.registerComponent(CLANG(self))
		self.registerComponent(VIROptimizer(self))
				
	def run(self):
		if self.clean:
			self.cleanIntermediateFiles()
			return
		
		self.compile()

	def cleanIntermediateFiles(self):
		if self.verbose:
			print "Removing intermediate files"
		
		for filename in self.intermediateFiles:
			if self.verbose:
				print " " + filename
			safeRemove(filename)
	
	def compile(self):
		self.compileFile(self.outputFile, self.inputFile)
		
		if self.verbose:
			print "Compilation Succeeded"			
		
	def compileFile(self, outputFile, inputFile):
	
		oldOutputFile    = self.outputFile
		oldIntermediates = self.intermediateFiles

		self.outputFile = outputFile
	
		self.computeIntermediateFiles(inputFile)

		currentFiles = [inputFile]
		
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
	
	def computeIntermediateFiles(self, inputFile):

		self.intermediateFiles = set([inputFile])

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
		
		self.intermediateFiles.remove(inputFile)
		
	def getCompilerThatCanHandle(self, files):
		for component in self.components:
			if component.canHandleAtLeastOne(files):
				return component
	
		return None

	def validateInputs(self):
		if len(self.inputFile) == 0:
			raise ValueError('No input file specified.')

		if len(self.outputFile) == 0:
			if self.onlyAssemble:
				self.outputFile = getBase(self.inputFile) + '.llvm'
			else:
				self.outputFile = getBase(self.inputFile) + '.bc'

	def loadLogger(self):
		self.logger = logging.getLogger('VanaheimrCompiler')
		logging.basicConfig()
		
		if self.verbose:
			self.logger.setLevel(logging.DEBUG)
		else:
			self.logger.setLevel(logging.ERROR)

		self.logger.info("Loaded knobs: " + str(self.knobs))

	def isKnobEnabled(self, knob):
		if not knob in self.knobs:
			return False
		
		return isTrue(self.knobs[knob])

	def getKnobValue(self, knob):
		if not knob in self.knobs:
			return None
		
		return self.knobs[knob]

def interpretKnobs(knobs):
	knobMap = {}
	
	for knob in knobs:
		splitKnob = knob.split('=')
		
		if len(splitKnob) == 1:
			splitKnob = knob.split(',')
			
		if len(splitKnob) == 1:
			splitKnob = knob.split('.')

		if len(splitKnob) == 1:
			continue

		name, value = splitKnob
		
		knobMap[name.strip()] = value.strip()

	return knobMap

def isTrue(string):
	if string == "True":
		return True
		
	if string == 1:
		return True
		
	if string == "1":
		return True

	if string == True:
		return True

	return False


