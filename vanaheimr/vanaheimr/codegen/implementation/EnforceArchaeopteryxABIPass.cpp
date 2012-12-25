/*! \file   EnforceArchaeopteryxABIPass.cpp
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Friday December 21, 2021
	\brief  The source file for the EnforceArchaeopteryxABIPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/EnforceArchaeopteryxABIPass.h>

namespace vanaheimr
{

namespace codegen
{

EnforceArchaeopteryxABIPass::EnforceArchaeopteryxABIPass()
: ModulePass({""}, "EnforceArchaeopteryxABIPass")
{

}

typedef util::LargeMap<std::string, uint64_t> GlobalToAddressMap;
typedef util::SmallMap<std::string, uint64_t>  LocalToAddressMap;

void layoutGlobals(ir::Module& module, GlobalToAddressMap& globals,
	const abi::ApplicationBinaryInterface& abi);
void layoutLocals(ir::Function& function, LocalToAddressMap& globals,
	const abi::ApplicationBinaryInterface& abi);
void lowerFunction(ir::Function& function,
	const abi::ApplicationBinaryInterface& abi,
	const GlobalToAddressMap& globals, const LocalToAddressMap& locals);

void EnforceArchaeopteryxABIPass::runOnModule(Module& m)
{
	GlobalToAddressMap globals;

	layoutGlobals(m, globals);
	
	// barrier
	
	// For-all
	for(auto function = m.begin(); function != m.end(); ++function)
	{
		LocalToAddressMap locals;
	
		layoutLocals(function, locals);
		
		// barrier

		lowerFunction(function, abi, globals, locals);
	}
}

void layoutGlobals(ir::Module& module, GlobalToAddressMap& globals)
{
	unsigned int offset = 0;

	for(auto global = module.global_begin();
		global != module.global_end(); ++global)
	{
		offset = align(offset, global->alignment());
		
		globals.insert(std::make_pair(global->name(), offset))

		offset += global->bytes();
	}
}

void layoutLocals(ir::Function& function, LocalToAddressMap& locals)
{
	// TODO
}

void lowerCall(ir::Instruction* i)
{
	assertM(false, "not implemented.");
}

void lowerReturn(ir::Instruction* i)
{
	assertM(false, "Not implemented.");
}

void lowerAddress(ir::Operand*& read, const GlobalToAddressMap& globals,
	const LocalToAddressMap& locals)
{
	auto variableRead = static_cast<ir::AddressOperand*>(read);

	auto local = locals.find(variableRead->globalValue->name());
	
	if(local != locals.end())
	{
		auto immediate = new ir::ImmediateOperand(local->second,
			read->instruction, read->type);

		read = immediate;

		delete variableRead;
		return;
	}

	auto global = globals.find(variableRead->globalValue->name());

	assert(global != globals.end());

	auto immediate = new ir::ImmediateOperand(global->second,
		read->instruction, read->type);

	read = immediate;

	delete variableRead;

}

void lowerEntryPoint(ir::Function& function, 
	const abi::ApplicationBinaryInterface& abi)
{
	// kernels don't need explicit entry point code
	if(function.hasAttribute("kernel")) return;

	assertM(false, "Entry point handling for called "
		"functions is not implemented yet");
}

void lowerFunction(ir::Function& function,
	const abi::ApplicationBinaryInterface& abi,
	const GlobalToAddressMap& globals, const LocalToAddressMap& locals)
{
	// add an entry point
	lowerEntryPoint(function, abi);

	// for all 
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		for(auto instruction : *block)
		{
			// lower calls
			if(instruction->isCall())
			{
				lowerCall(instruction, abi);
			}
			
			// lower returns
			if(instruction->isReturn())
			{
				lowerReturn(instruction, abi);
			} 

			// lower variable accesses
			for(auto read : instruction->reads)
			{
				if(read->isAddress())
				{
					lowerAddress(read, globals, locals);
				}
			}
		}
	}
}

}

}

