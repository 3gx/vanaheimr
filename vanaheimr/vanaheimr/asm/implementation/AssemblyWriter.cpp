/*! \file   AssemblyWriter.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday March 4, 2012
	\brief  The source file for the AssemblyWriter class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/AssemblyWriter.h>

namespace vanaheimr
{

namespace asm
{

AssemblyWriter::AssemblyWriter()
{

}

void AssemblyWriter::write(std::ostream& stream, const ir::Module& module)
{
	report("Writing assembly for module '" << module.name << "'");

	for(auto function : module)
	{
		writeFunction(stream, function);
	}
	
	for(auto global = module.global_begin();
		global != module.global_end(); ++global)
	{
		writeGlobal(stream, *global);
	}
}

void AssemblyWriter::writeFunction(std::ostream& stream,
	const ir::Function& function)
{
	report(" For function '" << function.name << "'");

	stream << ".function ";
	
	writeLinkage(stream, function); 
	
	stream << " " << function.name << "(";
	
	for(ir::Function::const_argument_iterator
		argument = function.argument_begin();
		argument != function.argument_end(); ++argument)
	{
		if(argument != function.argument_begin()) stream << ", ";

		writeArgument(stream, *argument);
	}
	
	stream << ")\n{\n";
	
	for(ir::Function::const_iterator block = function.begin();
		block != function.end(); ++block)
	{
		writeBasicBlock(stream, *block);
	}
	
	stream << "}\n";
}

void AssemblyWriter::writeGlobal(std::ostream& stream, const ir::Global& global)
{
	report(" For global '" << global.name << "'");
	
	stream << ".global ";
	
	writeLinkage(stream, global);
	writeType(stream, global);
	
	if(global.hasInitializer())
	{
		stream << " = ";
		writeInitializer(stream, global.initializer());
	}
}

void AssemblyWriter::writeLinkage(std::ostream& stream,
	const ir::Variable& variable)
{
	switch(variable.linkage())
	{
	case ir::Variable::ExternalLinkage:
	{
		stream << ".external ";
		break;
	}
	case ir::Variable::LinkOnceAnyLinkage:
	{
		stream << ".inline ";
		break;
	}
	case ir::Variable::LinkOnceODRLinkage:
	{
		stream << ".inline_strict ";
		break;
	}
	case ir::Variable::WeakAnyLinkage:
	{
		stream << ".weak ";
		break;
	}
	case ir::Variable::InternalLinkage:
	{
		stream << ".internal ";
		break;
	}
	case ir::Variable::PrivateLinkage:
	{
		stream << ".private ";
		break;
	}
	}
	
	switch(variable.visibility())
	{
	case ir::Variable::VisibleVisibility:
	{
		stream << ".visible ";
		break;
	}
	case ir::Variable::ProtectedVisibility:
	{
		stream << ".protected ";
		break;
	}
	case ir::Variable::HiddenVisibility:
	{
		stream << ".hidden ";
		break;
	}
	}
}

void AssemblyWriter::writeArgument(std::ostream& stream,
	const ir::Argument& argument)
{
	writeType(stream, argument.type());
	
	stream << " " << argument.name();
}

void AssemblyWriter::writeBasicBlock(std::ostream& stream,
	const ir::BasicBlock& block)
{
	stream << "\t" << block.name() << ":\n";
	
	for(auto instruction : block)
	{
		stream << "\t\t";
		writeOpcode(stream, instruction.opcode);
		
		
	}
}

void AssemblyWriter::writeType(std::ostream& stream, const ir::Type& type)
{
	if(type.isPrimitive())
	{
		if(type.isInteger())
		{
			const ir::IntegerType& integerType =
				static_cast<const ir::IntegerType&>(type);
				
			stream << ".i" << integerType.bits() << " ";
		}
		else if(type.isSinglePrecisionFloat())
		{
			stream << ".float ";
		}
		else if(type.isDoublePrevisionFloat())
		{
			stream << ".double ";
		}
		else
		{
			throw std::runtime_error("Invalid primitive type " + type.name());
		}
	}
	else
	{
		assertM(false, "Not implemented.");
	}
}

void AssemblyWriter::writeInitializer(std::ostream& stream,
	const ir::Constant& constant)
{
	assertM(false, "Not implemented.");
}

}

}

