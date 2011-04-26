/*! \file   CoreSimKernel.h
	\date   Saturday Feburary 23, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the Kernel class.
*/

#pragma once

/*! \brief A namespace for program execution */
namespace executive
{

class CoreSimKernel
{
    private:
        

    public:
        __device__ CoreSimKernel(*gpuState, file* binary);
        __device__ void launchKernel();

};

}
