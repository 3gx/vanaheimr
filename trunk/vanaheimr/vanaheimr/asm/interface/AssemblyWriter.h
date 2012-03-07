/*! \file   AssemblyWriter.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday March 4, 2012
	\brief  The header file for the AssemblyWriter class.
*/

#pragma once

// Forward Declarations
namespace vanaheimr{ namespace ir { class Module;     } }
namespace vanaheimr{ namespace ir { class Function;   } }
namespace vanaheimr{ namespace ir { class Global;     } }
namespace vanaheimr{ namespace ir { class Variable;   } }
namespace vanaheimr{ namespace ir { class Argument;   } }
namespace vanaheimr{ namespace ir { class BasicBlock; } }
namespace vanaheimr{ namespace ir { class Type;       } }
namespace vanaheimr{ namespace ir { class Constant;   } }

namespace vanaheimr
{

namespace asm
{

/*! \brief Used to write a module to an object file */
class AssemblyWriter
{
public:
	AssemblyWriter();

public:
	void write(std::ostream& stream, const ir::Module& m);

private:
	void writeFunction(std::ostream& stream, const ir::Function& f);
	void writeGlobal(std::ostream& stream, const ir::Global& g);

	void writeLinkage(std::ostream& stream, const ir::Variable& v);
	void writeArgument(std::ostream& stream, const ir::Argument& a);
	void writeBasicBlock(std::ostream& stream, const ir::BasicBlock& b);
	
	void writeType(std::ostream& stream, const ir::Type& t);
	void writeInitializer(std::ostream& stream, const ir::Constant& c);
};

}

}

