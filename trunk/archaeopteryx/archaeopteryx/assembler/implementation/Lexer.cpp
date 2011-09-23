/*! \file   Lexer.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday September 12, 2011
	\brief  The header file for the Lexer class.
*/

#pragma once

#define threads      128
#define ctas         120
#define transactions 6

/*! \brief A namespace for VIR assembler related classes and functions */
namespace assembler
{

__device__ Lexer::Lexer(util::File* file)
: _file(file), _fileData(0), _fileDataSize(0), _splitters(0),
	_transposedFileData(0), _transposedTokens(0), _tokens(0), _tokenCount(0)
{

}

__device__ Lexer::~Lexer()
{
	delete _file;
	delete[] _fileData;
	delete[] _splitters;
	delete[] _transposedFileData;
	delete[] _transposedTokens;
	delete[] _tokens;
}

__device__ const Lexer::Token* Lexer::tokenStream() const
{
	return _tokens;
}

__device__ size_t Lexer::tokenCount() const
{
	return _tokenCount;
}

__device__ void Lexer::lex()
{
	_fileDataSize       = _file->size();
	_fileData           = new char[_fileDataSize];
	_transposedFileData = new char[_fileDataSize];

	const unsigned int threadCount = ctas * threads;

	_splitters        = new Splitter[threadCount];
	_transposedTokens = new Token[_fileDataSize];
	_tokens           = new Token[_fileDataSize];
	
	util::HostReflection::launch(ctas, threads, "Lexer::findSplitters", this);
	util::HostReflection::launch(ctas, threads,
		"Lexer::transposeStreams", this);
	util::HostReflection::launch(ctas, threads,
		"Lexer::lexCharacterStreams", this);
	util::HostReflection::launch(ctas, threads,
		"Lexer::transposeCharacterStreams", this);
	util::HostReflection::launch(ctas, threads,
		"Lexer::gatherTokenStreams", this);
	util::HostReflection::launch(1, 1,
		"Lexer::cleanup", this);
}

__device__ void Lexer::findSplitters()
{
	size_t totalThreads = threads * ctas;
	size_t blockSize = _fileDataSize / totalThreads;
	size_t threadId  = util::threadId();
	
	char*  startingPoint = _fileData + (threadId + 1) * blockSize;
	char*  fileEnd       = _fileData + _fileDataSize;
	
	startingPoint = totalThreads - 1 == threadId ? fileEnd : startingPoint;
	
	for(; startingPoint < fileEnd; ++startingPoint)
	{
		if(*startingPoint == '\n') break;
	}
	
	_splitters[threadId] = startingPoint;
}

__device__ void Lexer::transposeStreams()
{
	Splitter range[transactions];
	
	 = _getSplitter();
	
	__shared__ char buffer[(transactions+1)*threads];
	
	
}

__device__ void Lexer::lexCharacterStreams();
__device__ void Lexer::transposeTokenStreams();
__device__ void Lexer::gatherTokenStreams();

__device__ void Lexer::cleanup()
{
	delete[] _fileData;
	_fileData = 0;
	
	delete[] _transposedFileData;
	_trabnsposedFileData = 0;

	delete[] _splitters; 
	_splitters = 0;

	delete[] _transposedFileData;
	_transposedFileData = 0;

	delete[] _transposedTokens;
	_transposedTokens = 0;
}

}

