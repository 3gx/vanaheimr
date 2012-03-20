/*! \file  SymbolTableEntry.h
	\date   Saturday March 4, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The header file for the specification of the symbol table of the binary
*/

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{
/*! \brief A namespace for the internal representation */
namespace as
{

class SymbolTableEntry
{
public:
    uint32_t type         : 32;
    uint32_t attributes   : 32;
    uint64_t stringOffset : 64;
    uint64_t offset       : 64;
};

}

}

