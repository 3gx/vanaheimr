/*! \file   CoreSimKernel.h
	\date   Sunday September 19, 2011
	\author Sudnya Padalikar
		<mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the Kernel class.
*/

#pragma once

/*! \brief A namespace for program execution */
//forward declarations
namespace executive {class CoreSimBlock;}

namespace executive
{

class CoreSimKernel
{
    private:
        

    public:
        //__device__ CoreSimKernel(void *gpuState, char* binaryName);
        __device__ void launchKernel(unsigned int simulatedBlocks, executive::CoreSimBlock* blocks);

};

}
