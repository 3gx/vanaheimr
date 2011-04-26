/*! \file   CoreSimThread.h
	\date   Saturday Feburary 23, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the thread class.
*/

#pragma once

/*! \brief A namespace for program execution */
namespace executive
{
class CoreSimThread
{
    private:
        
    public:
        __device__ CoreSimThread(binary*, threadid, regsused);
        __device__ PC executeInstruction(instr*, PC);
};

}

