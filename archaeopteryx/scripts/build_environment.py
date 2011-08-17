EnsureSConsVersion(1,2)

import os

import inspect
import platform
import re
import subprocess
from SCons import SConf

def getCudaPaths():
	"""Determines CUDA {bin,lib,include} paths
	
	returns (bin_path,lib_path,inc_path)
	"""

	# determine defaults
	if os.name == 'nt':
		bin_path = 'C:/CUDA/bin'
		lib_path = 'C:/CUDA/lib'
		inc_path = 'C:/CUDA/include'
	elif os.name == 'posix':
		bin_path = '/usr/local/cuda/bin'
		lib_path = '/usr/local/cuda/lib'
		inc_path = '/usr/local/cuda/include'
	else:
		raise ValueError, 'Error: unknown OS.  Where is nvcc installed?'
	 
	if platform.machine()[-2:] == '64':
		lib_path += '64'

	# override with environement variables
	if 'CUDA_BIN_PATH' in os.environ:
		bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
	if 'CUDA_LIB_PATH' in os.environ:
		lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
	if 'CUDA_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])

	return (bin_path,lib_path,inc_path)

def getBoostPaths():
	"""Determines BOOST {bin,lib,include} paths
	
	returns (bin_path,lib_path,inc_path)
	"""

	# determine defaults
	if os.name == 'posix':
		bin_path = '/usr/bin'
		lib_path = '/usr/lib'
		inc_path = '/usr/include'
	else:
		raise ValueError, 'Error: unknown OS.  Where is boost installed?'

	# override with environement variables
	if 'BOOST_BIN_PATH' in os.environ:
		bin_path = os.path.abspath(os.environ['BOOST_BIN_PATH'])
	if 'BOOST_LIB_PATH' in os.environ:
		lib_path = os.path.abspath(os.environ['BOOST_LIB_PATH'])
	if 'BOOST_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['BOOST_INC_PATH'])

	return (bin_path,lib_path,inc_path)

def getFlexPaths(env):
	"""Determines Flex {include} paths

	returns (inc_path)
	"""

	# determine defaults
	if os.name == 'posix':
		inc_path = '/usr/include'
	elif os.name == 'nt':
		inc_path = 'C:\MinGW\1.0'
	else:
		raise ValueError, 'Error: unknown OS.  Where is GLEW installed?'

	# override with environement variables
	if 'FLEX_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['GLEW_INC_PATH'])

	return (inc_path)

def getGLEWPaths(env):
	"""Determines GLEW {bin,lib,include} paths and is it installed?

	returns (have_glew,bin_path,lib_path,inc_path)
	"""

	configure = Configure(env)
	glew = configure.CheckLib('GLEW')		
	
	if not glew:
		return (glew, '', '', '')

	# determine defaults
	if os.name == 'posix':
		bin_path = '/usr/bin'
		lib_path = '/usr/lib'
		inc_path = '/usr/include'
	else:
		raise ValueError, 'Error: unknown OS.  Where is GLEW installed?'

	# override with environement variables
	if 'GLEW_BIN_PATH' in os.environ:
		bin_path = os.path.abspath(os.environ['GLEW_BIN_PATH'])
	if 'GLEW_LIB_PATH' in os.environ:
		lib_path = os.path.abspath(os.environ['GLEW_LIB_PATH'])
	if 'GLEW_INC_PATH' in os.environ:
		inc_path = os.path.abspath(os.environ['GLEW_INC_PATH'])

	return (glew,bin_path,lib_path,inc_path)

def getLLVMPaths(enabled):
	"""Determines LLVM {have,bin,lib,include,cflags,lflags,libs} paths
	
	returns (have,bin_path,lib_path,inc_path,cflags,lflags,libs)
	"""
	
	if not enabled:
		return (False, [], [], [], [], [], [])
	
	try:
		llvm_config_path = which('llvm-config')
	except:
		print 'Failed to find llvm-config'
		return (False, [], [], [], [], [], [])
	
	# determine defaults
	if os.name == 'posix':
		bin_path = os.popen('llvm-config --bindir').read().split()
		lib_path = os.popen('llvm-config --libdir').read().split()
		inc_path = os.popen('llvm-config --includedir').read().split()
		cflags   = os.popen('llvm-config --cppflags').read().split()
		lflags   = os.popen('llvm-config --ldflags').read().split()
		libs     = os.popen('llvm-config --libs core jit native \
			asmparser instcombine').read().split()
	else:
		raise ValueError, 'Error: unknown OS.  Where is LLVM installed?'
	
	# remove -DNDEBUG
	if '-DNDEBUG' in cflags:
		cflags.remove('-DNDEBUG')

	# remove lib_path from libs
	for lib in libs:
		if lib[0:2] == "-L":
			libs.remove(lib)

	return (True,bin_path,lib_path,inc_path,cflags,lflags,libs)
	
def getTools():
	result = []
	if os.name == 'posix':
		result = ['default', 'c++', 'g++']
	else:
		result = ['default']

	return result;

OldEnvironment = Environment;

# this dictionary maps the name of a compiler program to a dictionary mapping the name of
# a compiler switch of interest to the specific switch implementing the feature
gCompilerOptions = {
		'gcc' : {'warn_all' : '-Wall', 'warn_errors' : '-Werror',
			'optimization' : '-O2', 'debug' : '-g', 
			'exception_handling' : '', 'standard': ''},
		'g++' : {'warn_all' : '-Wall', 'warn_errors' : '-Werror',
			'optimization' : '-O2', 'debug' : '-g', 
			'exception_handling' : '', 'standard': '-std=c++0x'}
	}


# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
		'gcc'  : {'debug' : ''},
		'g++'  : {'debug' : ''}
	}
	
def getCFLAGS(mode, warn, warnings_as_errors, CC):
	result = []
	if mode == 'release':
		# turn on optimization
		result.append(gCompilerOptions[CC]['optimization'])
	elif mode == 'debug':
		# turn on debug mode
		result.append(gCompilerOptions[CC]['debug'])
		result.append('-DOCELOT_DEBUG')
	
	if warn:
		# turn on all warnings
		result.append(gCompilerOptions[CC]['warn_all'])
	
	if warnings_as_errors:
		# treat warnings as errors
		result.append(gCompilerOptions[CC]['warn_errors'])
	
	result.append(gCompilerOptions[CC]['standard'])

	return result

def getCXXFLAGS(mode, warn, warnings_as_errors, CXX):
	result = []
	if mode == 'release':
		# turn on optimization
		result.append(gCompilerOptions[CXX]['optimization'])
	elif mode == 'debug':
		# turn on debug mode
		result.append(gCompilerOptions[CXX]['debug'])
	# enable exception handling
	result.append(gCompilerOptions[CXX]['exception_handling'])

	if warn:
		# turn on all warnings
		result.append(gCompilerOptions[CXX]['warn_all'])

	if warnings_as_errors:
		# treat warnings as errors
		result.append(gCompilerOptions[CXX]['warn_errors'])

	result.append(gCompilerOptions[CXX]['standard'])

	return result

def getLINKFLAGS(mode, LINK):
	result = []
	if mode == 'debug':
		# turn on debug mode
		result.append(gLinkerOptions[LINK]['debug'])

	return result

def importEnvironment():
	env = {  }
	
	if 'PATH' in os.environ:
		env['PATH'] = os.environ['PATH']
	
	if 'CXX' in os.environ:
		env['CXX'] = os.environ['CXX']
	
	if 'CC' in os.environ:
		env['CC'] = os.environ['CC']
	
	if 'LD_LIBRARY_PATH' in os.environ:
		env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']

	return env

def Environment():
	vars = Variables()

	# add a variable to handle RELEASE/DEBUG mode
	vars.Add(EnumVariable('mode', 'Release versus debug mode', 'release',
		allowed_values = ('release', 'debug')))

	# add a variable to handle warnings
	vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 1))

	# add a variable to treat warnings as errors
	vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 1))
	
	# add a variable to compile the unit tests
	vars.Add(EnumVariable('test_level',
		'Build the unit tests at the given test level', 'full',
		allowed_values = ('none', 'basic', 'full')))

	# add a variable to determine the install path
	vars.Add(PathVariable('install_path', 'The archaeopteryx install path',
		'/usr/local'))

	# create an Environment
	env = OldEnvironment(ENV = importEnvironment(),
		tools = getTools(), variables = vars)

	# get the absolute path to the directory containing
	# this source file
	thisFile = inspect.getabsfile(Environment)
	thisDir = os.path.dirname(thisFile)

	# get C compiler switches
	env.AppendUnique(CFLAGS = getCFLAGS(env['mode'], env['Wall'],
		env['Werror'], env.subst('$CC')))

	# get NVCC compiler switches
	env.Append(NVCCFLAGS = getNVCCFLAGS(env['mode'],
		env['backend'], env['arch']))

	# get CXX compiler switches
	env.AppendUnique(CXXFLAGS = getCXXFLAGS(env['mode'], env['Wall'],
		env['Werror'], env.subst('$CXX')))

	# get linker switches
	env.AppendUnique(LINKFLAGS = getLINKFLAGS(env['mode'], env.subst('$LINK')))

	# generate help text
	Help(vars.GenerateHelpText(env))

	return env

