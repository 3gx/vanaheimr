/*! \file   BinaryReader.cpp
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the BinaryReader class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryReader.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>
#include <unordered_set>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace as
{

ir::Module* BinaryReader::read(std::istream& stream, const std::string& name)
{
	_readHeader(stream);
	_readDataSection(stream);
	_readStringTable(stream);
	_readSymbolTable(stream);
	_readInstructions(stream);

	ir::Module* module = new ir::Module(name,
		compiler::Compiler::getSingleton());

	_loadTypes();
	_initializeModule(*module);
	
	return module;
}

void BinaryReader::_readHeader(std::istream& stream)
{
	report("Reading header...");
	stream.read((char*)&_header, sizeof(BinaryHeader));

	if(stream.gcount() != sizeof(BinaryHeader))
	{
		throw std::runtime_error("Failed to read binary "
			"header, hit EOF.");
	}

	report(" data pages:    " << _header.dataPages);
	report(" code pages:    " << _header.codePages);
	report(" symbols:       " << _header.symbols);
	report(" string pages:  " << _header.stringPages);
	report(" data offset:   " << _header.dataOffset);
	report(" code offset:   " << _header.codeOffset);
	report(" symbol offset: " << _header.symbolOffset);
	report(" string offset: " << _header.stringsOffset);
	report(" name offset:   " << _header.nameOffset);
}

void BinaryReader::_readDataSection(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.dataPages;

	stream.seekg(_header.dataOffset, std::ios::beg);

	_dataSection.resize(dataSize);

	stream.read((char*) _dataSection.data(), dataSize);

	if((size_t)stream.gcount() != dataSize)
	{
		throw std::runtime_error("Failed to read binary data section, hit"
			" EOF."); 
	}
}

void BinaryReader::_readStringTable(std::istream& stream)
{
	size_t stringTableSize = BinaryHeader::PageSize * _header.stringPages;

	stream.seekg(_header.stringsOffset, std::ios::beg);

	_stringTable.resize(stringTableSize);

	stream.read((char*) _stringTable.data(), stringTableSize);

	if((size_t)stream.gcount() != stringTableSize)
	{
		throw std::runtime_error("Failed to read string table, hit EOF");
	}
}

void BinaryReader::_readSymbolTable(std::istream& stream)
{
	size_t symbolTableSize = sizeof(SymbolTableEntry) * _header.symbols;

	stream.seekg(_header.symbolOffset, std::ios::beg);

	_symbolTable.resize(_header.symbols);

	stream.read((char*) _symbolTable.data(), symbolTableSize);

	if((size_t)stream.gcount() != symbolTableSize)
	{
		throw std::runtime_error("Failed to read symbol table, hit EOF");
	}
}

void BinaryReader::_readInstructions(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.codePages;
	size_t sizeInInstructions = (dataSize + sizeof(InstructionContainer) - 1) /
		sizeof(InstructionContainer);

	_instructions.resize(sizeInInstructions);

	// TODO obey page alignment
	stream.seekg(_header.codeOffset, std::ios::beg);

	stream.read((char*) _instructions.data(), dataSize);

	if((size_t)stream.gcount() != dataSize)
	{
		throw std::runtime_error("Failed to read code section, hit EOF.");
	}
}

void BinaryReader::_loadTypes()
{
	for(auto symbol = _symbolTable.begin();
		symbol != _symbolTable.end(); ++symbol)
	{
		compiler::Compiler::getSingleton()->getOrInsertType(
			_getSymbolTypeName(*symbol));
	}
}

void BinaryReader::_initializeModule(ir::Module& m)
{
	_loadGlobals(m);
	_loadFunctions(m);

	_variables.clear();
}

void BinaryReader::_loadGlobals(ir::Module& m)
{
	report(" Loading global variables from symbol table...");
	
	for(auto symbol = _symbolTable.begin();
		symbol != _symbolTable.end(); ++symbol)
	{
		if(symbol->type != SymbolTableEntry::VariableType &&
			symbol->type != SymbolTableEntry::FunctionType) continue;

		uint64_t symbolTableOffset = _header.symbolOffset +
			sizeof(SymbolTableEntry) *
			std::distance(_symbolTable.begin(), symbol);
		
		report("  loaded " << _getSymbolName(*symbol)
			<< " at offset " << symbol->offset
			<< ", symbol offset is " << symbolTableOffset);

		auto type = _getSymbolType(*symbol);

		if(type == nullptr)
		{
			throw std::runtime_error("Could not find type with name '" +
				_getSymbolTypeName(*symbol) + "' for symbol '" +
				_getSymbolName(*symbol) + "'");
		}

		ir::Variable* variable = nullptr;

		if(symbol->type == SymbolTableEntry::VariableType)
		{
			auto global = m.newGlobal(_getSymbolName(*symbol), type,
				_getSymbolLinkage(*symbol), _getSymbolLevel(*symbol));

			if(_hasInitializer(*symbol))
			{
				global->setInitializer(_getInitializer(*symbol));
			}
			
			variable = &*global;
		}
		else
		{
			ir::Module::iterator function = m.newFunction(
				_getSymbolName(*symbol), _getSymbolLinkage(*symbol),
				_getSymbolVisibility(*symbol));

			report("   loading arguments...");

			for(auto argumentSymbol = _symbolTable.begin();
				argumentSymbol != _symbolTable.end(); ++argumentSymbol)
			{
				if(argumentSymbol->type != SymbolTableEntry::ArgumentType)
				{
					continue;
				}
				
				std::string functionName =
					_getSymbolName(*argumentSymbol).substr(2,
					function->name().size());
			
				if(functionName != function->name()) continue;

				uint64_t symbolTableOffset = _header.symbolOffset +
					sizeof(SymbolTableEntry) *
					std::distance(_symbolTable.begin(), argumentSymbol);

				std::string name = _getSymbolName(*argumentSymbol).substr(
					2 + function->name().size());

				report("    loaded argument " << name
					<< " at offset " << argumentSymbol->offset
					<< ", symbol offset is " << symbolTableOffset);

				auto type = _getSymbolType(*argumentSymbol);

				if(type == nullptr)
				{
					throw std::runtime_error("Could not find type with name '" +
						_getSymbolTypeName(*argumentSymbol) + "' for symbol '" +
						name + "'");
				}
		
				auto argument = function->newArgument(type, name);

				_arguments.insert(std::make_pair(symbolTableOffset,
				&*argument));
			}
			
			variable = &*function;
		}
		
		_variables.insert(std::make_pair(symbolTableOffset, variable));
	}
}

void BinaryReader::_loadFunctions(ir::Module& m)
{
	typedef std::unordered_map<uint64_t, ir::BasicBlock*> PCToBasicBlockMap;

	report(" Loading functions from symbol table...");

	for(auto symbol = _symbolTable.begin();
		symbol != _symbolTable.end(); ++symbol)
	{
		if(symbol->type != SymbolTableEntry::FunctionType) continue;

		uint64_t symbolTableOffset = _header.symbolOffset +
			sizeof(SymbolTableEntry) *
			std::distance(_symbolTable.begin(), symbol);
	
		report("  loaded function " << _getSymbolName(*symbol));

		ir::Variable* variable = _getVariableAtSymbolOffset(symbolTableOffset);
		ir::Function* function = static_cast<ir::Function*>(variable);

		BasicBlockDescriptorVector blocks = _getBasicBlocksInFunction(*symbol);
	
		PCToBasicBlockMap blockPCs;

		for(auto blockOffset : blocks)
		{
			ir::Function::iterator block = function->newBasicBlock(
				function->end(), blockOffset.name);

			blockPCs.insert(std::make_pair(blockOffset.begin, &*block));

			report("   adding basic block using instructions [" 
				<< blockOffset.begin << ", " << blockOffset.end << "]");
		
			for(unsigned int i = blockOffset.begin; i != blockOffset.end; ++i)
			{
				assert(i < _instructions.size());
				_addInstruction(block, _instructions[i]);
				report("    added instruction '" 
					<< block->back()->toString() << "'");
			}
		}

		report("  resolving branch targets...");

		for(auto unresolved : _unresolvedTargets)
		{
			// find the symbol with the specified offset
			uint64_t symbolOffset = (unresolved.first -
				_header.symbolOffset) / sizeof(SymbolTableEntry);

			assert(symbolOffset < _symbolTable.size());

			const SymbolTableEntry& targetSymbol = _symbolTable[symbolOffset];

			uint64_t pc = (targetSymbol.offset - _header.codeOffset) /
				sizeof(InstructionContainer);
		
			report("   for branch to pc " << pc);

			auto block = blockPCs.find(pc);

			if(block == blockPCs.end())
			{
				std::stringstream message;

				message << "Could not find basic block starting at pc " << pc;

				throw std::runtime_error(message.str());
			}
			
			auto branch = static_cast<ir::Bra*>(unresolved.second);
			
			report("    setting target to " << block->second->name());

			static_cast<ir::AddressOperand*>(branch->target)->globalValue =
				block->second;
		}

		_unresolvedTargets.clear();
		_virtualRegisters.clear();
		_arguments.clear();
	}
}

std::string BinaryReader::_getSymbolName(const SymbolTableEntry& symbol) const
{
	return std::string((char*)_stringTable.data() + symbol.stringOffset);
}

std::string BinaryReader::_getSymbolTypeName(
	const SymbolTableEntry& symbol) const
{
	return std::string((char*)_stringTable.data() + symbol.typeOffset);
}

ir::Type* BinaryReader::_getSymbolType(const SymbolTableEntry& symbol) const
{
	return compiler::Compiler::getSingleton()->getType(
		_getSymbolTypeName(symbol));
}

ir::Variable::Linkage BinaryReader::_getSymbolLinkage(
	const SymbolTableEntry& symbol) const
{
	return (ir::Variable::Linkage)(symbol.attributes.linkage);
}

ir::Variable::Visibility BinaryReader::_getSymbolVisibility(
	const SymbolTableEntry& symbol) const
{
	return (ir::Variable::Visibility)(symbol.attributes.visibility);
}

ir::Global::Level BinaryReader::_getSymbolLevel(
	const SymbolTableEntry& symbol) const
{
	return (ir::Global::Level)symbol.attributes.level;
}

bool BinaryReader::_hasInitializer(const SymbolTableEntry& symbol) const
{
	// currently binaries never have initializers
	return false;
}

ir::Constant* BinaryReader::_getInitializer(
	const SymbolTableEntry& symbol) const
{
	assertM(false, "Not imeplemented.");
}

BinaryReader::BasicBlockDescriptorVector
	BinaryReader::_getBasicBlocksInFunction(
	const SymbolTableEntry& symbol) const
{
	typedef std::unordered_set<uint64_t> TargetSet;

	BasicBlockDescriptorVector blocks;
	
	// Get the first and last instruction in the function
	uint64_t begin = (symbol.offset - _header.codeOffset) /
		sizeof(InstructionContainer);
	
	uint64_t end = begin + symbol.size / sizeof(InstructionContainer);

	report("   getting basic block for symbol '" << _getSymbolName(symbol)
		<< "' (offset " << symbol.offset
		<< ", size " << symbol.size << ", range ["
		<< begin << ", " << end <<"])");

	TargetSet targets;

	for(uint64_t i = begin; i != end; ++i)
	{
		const archaeopteryx::ir::InstructionContainer&
			instruction = _instructions[i];

		if(instruction.asInstruction.opcode ==
			archaeopteryx::ir::Instruction::Bra)
		{
			report("   found branch at pc " << i);
			auto operand = instruction.asBra.target;

			if(operand.asOperand.mode == archaeopteryx::ir::Operand::Immediate)
			{
				report("    to target " << operand.asImmediate.uint);

				targets.insert(operand.asImmediate.uint);
			}
			else if(operand.asOperand.mode ==
				archaeopteryx::ir::Operand::Symbol)
			{
				uint64_t symbolOffset = (operand.asSymbol.symbolTableOffset -
					_header.symbolOffset) / sizeof(SymbolTableEntry);

				assert(symbolOffset < _symbolTable.size());

				const SymbolTableEntry& targetSymbol =
					_symbolTable[symbolOffset];
				
				uint64_t targetPC = (targetSymbol.offset - _header.codeOffset) /
					sizeof(InstructionContainer);
				
				report("    to target " << targetPC);

				targets.insert(targetPC);
			}
			else
			{
				assertM(false, "branch mode "
					<< operand.asOperand.mode << " not implemented");
			}
		}
	}

	BasicBlockDescriptor block("BB_0", begin);

	for(uint64_t i = begin; i != end; ++i)
	{
		bool isTerminator = false;
		uint64_t blockEnd = i;

		const archaeopteryx::ir::InstructionContainer&
			instruction = _instructions[i];

		if(targets.count(i) != 0)
		{
			isTerminator = true;
		}
		else if(instruction.asInstruction.opcode ==
			archaeopteryx::ir::Instruction::Bra)
		{
			isTerminator = true;
			blockEnd = i + 1;
		}

		if(isTerminator)
		{
			block.end = blockEnd;
			blocks.push_back(block);

			std::stringstream name;

			name << "BB_" << blocks.size();

			block = BasicBlockDescriptor(name.str(), blockEnd);
		}
	}

	if(block.begin != end)
	{
		block.end = end;
		blocks.push_back(block);
	}

	return blocks;
}

void BinaryReader::_addInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	if(_addSimpleBinaryInstruction(block, container)) return;
	if(_addSimpleUnaryInstruction(block, container))  return;
	if(_addComplexInstruction(block, container))      return;

	assertM(false, "Translation for instruction '" <<
		ir::Instruction::toString((ir::Instruction::Opcode)
		container.asInstruction.opcode) << "' not implemented.");
}

bool BinaryReader::_addSimpleBinaryInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{	
	switch(container.asInstruction.opcode)
	{
	case archaeopteryx::ir::Instruction::Add:
	case archaeopteryx::ir::Instruction::And:
	case archaeopteryx::ir::Instruction::Ashr:
	case archaeopteryx::ir::Instruction::Fdiv:
	case archaeopteryx::ir::Instruction::Fmul:
	case archaeopteryx::ir::Instruction::Frem:
	case archaeopteryx::ir::Instruction::Lshr:
	case archaeopteryx::ir::Instruction::Mul:
	case archaeopteryx::ir::Instruction::Or:
	case archaeopteryx::ir::Instruction::Sdiv:
	case archaeopteryx::ir::Instruction::Shl:
	case archaeopteryx::ir::Instruction::Srem:
	case archaeopteryx::ir::Instruction::Sub:
	case archaeopteryx::ir::Instruction::Udiv:
	case archaeopteryx::ir::Instruction::Urem:
	case archaeopteryx::ir::Instruction::Xor:
	{
		auto instruction = static_cast<ir::BinaryInstruction*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asBinaryInstruction.guard, instruction));

		instruction->setD(_translateOperand(container.asBinaryInstruction.d,
			instruction));
		instruction->setA(_translateOperand(container.asBinaryInstruction.a,
			instruction));
		instruction->setB(_translateOperand(container.asBinaryInstruction.b,
			instruction));

		block->push_back(instruction);
	
		return true;
	}
	default: break;
	}

	return false;
}

bool BinaryReader::_addSimpleUnaryInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	switch(container.asInstruction.opcode)
	{
	case archaeopteryx::ir::Instruction::Bitcast:
	case archaeopteryx::ir::Instruction::Fpext:
	case archaeopteryx::ir::Instruction::Fptosi:
	case archaeopteryx::ir::Instruction::Fptoui:
	case archaeopteryx::ir::Instruction::Fptrunc:
	case archaeopteryx::ir::Instruction::Ld:
	case archaeopteryx::ir::Instruction::Sext:
	case archaeopteryx::ir::Instruction::Sitofp:
	case archaeopteryx::ir::Instruction::Trunc:
	case archaeopteryx::ir::Instruction::Uitofp:
	case archaeopteryx::ir::Instruction::Zext:
	{
		auto instruction = static_cast<ir::UnaryInstruction*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asUnaryInstruction.guard, instruction));

		instruction->setD(_translateOperand(container.asUnaryInstruction.d,
			instruction));
		instruction->setA(_translateOperand(container.asUnaryInstruction.a,
			instruction));

		block->push_back(instruction);
		
		return true;
	}
	default: break;
	}

	return false;
}

bool BinaryReader::_addComplexInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	if(container.asInstruction.opcode == archaeopteryx::ir::Instruction::St)
	{
		auto instruction = static_cast<ir::St*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asSt.guard, instruction));

		instruction->setD(_translateOperand(container.asSt.d,
			instruction));
		instruction->setA(_translateOperand(container.asSt.a,
			instruction));

		block->push_back(instruction);
		
		return true;
		
	}
	else if(container.asInstruction.opcode
		== archaeopteryx::ir::Instruction::Setp)
	{
		auto instruction = static_cast<ir::Setp*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asSetP.guard, instruction));

		instruction->setD(_translateOperand(container.asSetP.d,
			instruction));
		instruction->setA(_translateOperand(container.asSetP.a,
			instruction));
		instruction->setB(_translateOperand(container.asSetP.b,
			instruction));

		instruction->comparison =
			(ir::ComparisonInstruction::Comparison)
			container.asSetP.comparison;

		block->push_back(instruction);
		
		return true;
		
	}
	else if(container.asInstruction.opcode
		== archaeopteryx::ir::Instruction::Bra)
	{
		auto instruction = static_cast<ir::Bra*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asBra.guard, instruction));

		instruction->setTarget(_translateOperand(container.asBra.target,
			instruction));
		
		instruction->modifier = (ir::Bra::BranchModifier)
			container.asBra.modifier;

		block->push_back(instruction);
		
		return true;
	}
	else if(container.asInstruction.opcode
		== archaeopteryx::ir::Instruction::Call)
	{
		_addCallInstruction(block, container);

		return true;
	}
	else if(container.asInstruction.opcode
		== archaeopteryx::ir::Instruction::Ret)
	{
		auto instruction = static_cast<ir::Ret*>(
			ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

		instruction->setGuard(_translateOperand(
			container.asRet.guard, instruction));

		block->push_back(instruction);
		
		return true;
	}
	
	return false;
}

void BinaryReader::_addCallInstruction(ir::Function::iterator block,
	const InstructionContainer& container)
{
	auto instruction = static_cast<ir::Call*>(
		ir::Instruction::create((ir::Instruction::Opcode)
			container.asInstruction.opcode, &*block));

	instruction->setGuard(_translateOperand(
		container.asCall.guard, instruction));

	instruction->setTarget(_translateOperand(container.asCall.target,
		instruction));
	
	for(unsigned int returned = 0;
		returned != container.asCall.returnArguments; ++returned)
	{
		uint64_t offset = returned * sizeof(OperandContainer) +
			container.asCall.returnArgumentOffset;
		const OperandContainer* operand =
			reinterpret_cast<OperandContainer*>(&_dataSection[offset]);
	
		instruction->addReturn(_translateOperand(*operand, instruction));
	}

	for(unsigned int argument = 0;
		argument != container.asCall.arguments; ++argument)
	{
		uint64_t offset = argument * sizeof(OperandContainer) +
			container.asCall.argumentOffset;
		const OperandContainer* operand =
			reinterpret_cast<OperandContainer*>(&_dataSection[offset]);
	
		instruction->addArgument(_translateOperand(*operand, instruction));
	}
	
	block->push_back(instruction);
}

ir::Operand* BinaryReader::_translateOperand(const OperandContainer& container,
	ir::Instruction* instruction)
{
	typedef archaeopteryx::ir::Operand Operand;

	switch(container.asOperand.mode)
	{
	case Operand::Predicate:
	{
		return _translateOperand(container.asPredicate, instruction);
	}
	case Operand::Register:
	{
		ir::RegisterOperand* operand = new ir::RegisterOperand(
			_getVirtualRegister(container.asRegister.reg,
			container.asRegister.type, instruction->block->function()),
			instruction);
		
		return operand;
	}
	case Operand::Immediate:
	{
		ir::ImmediateOperand* operand = new ir::ImmediateOperand(
			container.asImmediate.uint, instruction,
			_getType(container.asImmediate.type));

		return operand;
	}
	case Operand::Indirect:
	{
		ir::IndirectOperand* operand = new ir::IndirectOperand(
			_getVirtualRegister(container.asIndirect.reg,
			container.asIndirect.type, instruction->block->function()),
			container.asIndirect.offset, instruction);
		
		return operand;
	}
	case Operand::Symbol:
	{
		ir::Argument* argument = _getArgumentAtSymbolOffset(
			container.asSymbol.symbolTableOffset);

		if(argument != nullptr)
		{
			ir::ArgumentOperand* operand = new ir::ArgumentOperand(argument,
				instruction);
			
			return operand;
		}

		ir::Variable* variable = _getVariableAtSymbolOffset(
			container.asSymbol.symbolTableOffset);

		ir::AddressOperand* operand = new ir::AddressOperand(
			variable, instruction);
	
		if(variable == nullptr)
		{
			_unresolvedTargets.insert(std::make_pair(
				container.asSymbol.symbolTableOffset, instruction));
		}

		return operand;
	}
	case Operand::InvalidOperand: break;
	}	

	assertM(false, "Invalid operand type.");

	return 0;
}

ir::PredicateOperand* BinaryReader::_translateOperand(
	const PredicateOperand& operand, ir::Instruction* instruction)
{
	ir::VirtualRegister* virtualRegister = nullptr;

	if(operand.modifier != archaeopteryx::ir::PredicateOperand::PredicateTrue &&
		operand.modifier != archaeopteryx::ir::PredicateOperand::PredicateFalse)
	{
		virtualRegister = _getVirtualRegister(operand.reg,
			archaeopteryx::ir::i1, instruction->block->function());
	}

	return new ir::PredicateOperand(virtualRegister, 
		(ir::PredicateOperand::PredicateModifier)operand.mode,
		instruction);
}

const ir::Type* BinaryReader::_getType(archaeopteryx::ir::DataType type) const
{
	switch(type)
	{
	case archaeopteryx::ir::i1:
	{
		return compiler::Compiler::getSingleton()->getType("i1");
	}
	case archaeopteryx::ir::i8:
	{
		return compiler::Compiler::getSingleton()->getType("i8");
	}
	case archaeopteryx::ir::i16:
	{
		return compiler::Compiler::getSingleton()->getType("i16");
	}
	case archaeopteryx::ir::i32:
	{
		return compiler::Compiler::getSingleton()->getType("i32");
	}
	case archaeopteryx::ir::i64:
	{
		return compiler::Compiler::getSingleton()->getType("i64");
	}
	case archaeopteryx::ir::f32:
	{
		return compiler::Compiler::getSingleton()->getType("f32");
	}
	case archaeopteryx::ir::f64:
	{
		return compiler::Compiler::getSingleton()->getType("f64");
	}
	default: break;
	}

	assertM(false, "Invalid data type.");

	return 0;
}
	
ir::VirtualRegister* BinaryReader::_getVirtualRegister(
	archaeopteryx::ir::RegisterType reg,
	archaeopteryx::ir::DataType type, ir::Function* function)
{
	auto virtualRegister = _virtualRegisters.find(reg);

	if(_virtualRegisters.end() == virtualRegister)
	{
		std::stringstream name;

		name << "r" << reg;

		auto insertedRegister = function->newVirtualRegister(
			_getType(type), name.str());
	
		virtualRegister = _virtualRegisters.insert(std::make_pair(reg,
			&*insertedRegister)).first;
	}

	return virtualRegister->second;
}

ir::Variable* BinaryReader::_getVariableAtSymbolOffset(uint64_t offset) const
{
	auto variable = _variables.find(offset);

	if(variable == _variables.end())
	{
		return nullptr;
	}

	return variable->second;
}

ir::Argument* BinaryReader::_getArgumentAtSymbolOffset(uint64_t offset) const
{
	auto argument = _arguments.find(offset);

	if(argument == _arguments.end())
	{
		return nullptr;
	}

	return argument->second;
}

BinaryReader::BasicBlockDescriptor::BasicBlockDescriptor(
	const std::string& n, uint64_t b, uint64_t e)
: name(n), begin(b), end(e)
{

}

}

}

