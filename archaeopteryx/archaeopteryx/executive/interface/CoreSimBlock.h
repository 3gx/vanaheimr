/*! \file   CoreSimBlock.h
	\date   Saturday Feburary 23, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the thread block class.
*/

#pragma once

//Forward declarations
namespace executive { class BlockState; }
namespace ir        { class Binary; }
/*! \brief A namespace for program execution */
namespace executive
{
class CoreSimBlock
{
    private:
        //FetchUnit m_fetchUnit;

    public:
        __device__ CoreSimBlock(BlockState* blockState, ir::Binary* binary);
        __device__ CoreSimThread* getCoreSimThread(unsigned int id);
        __device__ unsigned int getSimulatedThreadCount();
        __device__ CoreSimThread::Value getRegister(unsigned int, unsigned int);
        __device__ void setRegister(unsigned int, unsigned int, const CoreSimThread::Value&);
        __device__ CoreSimThread::Value translateVirtualToPhysical(const CoreSimThread::Value);
        __device__ void barrier(unsigned int);
        __device__ unsigned int returned(unsigned int, unsigned int);
};

}

