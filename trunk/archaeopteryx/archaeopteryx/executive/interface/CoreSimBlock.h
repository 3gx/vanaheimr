/*! \file   CoreSimBlock.h
	\date   Saturday Feburary 23, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the thread block class.
*/

#pragma once

/*! \brief A namespace for program execution */
namespace executive
{
class CoreSimBlock
{
    private:
        FetchUnit m_fetchUnit;

    public:
        __device__ CoreSimBlock(*blockState, Binary* binary);
        __device__ CoreSimThread* getCoreSimThread(unsigned int id);
        __device__ unsigned int getSimulatedThreadCount();
};

}

