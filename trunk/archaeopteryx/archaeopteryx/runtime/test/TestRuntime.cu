/*! \file TestRuntime.cu
 *  \date   Tuesday September 27, 2011
 *  \author Sudnya Padalikar
 *  <mailsudnya@gmail.com>
 *  \brief  A test file to test the implementation of Runtime.
 *   */

#include <archaeopteryx/runtime/interface/Runtime.h>
// Create i/p data, load binary, launch threads/run kernel, read outputs and verify outputs 
// allocate memory for i/p data
// maybe add memset/memcpy in runtime?
// Runtime loads binary from a given filename 
// Runtime finds the entry pt function
// simulation is launched by kernel for that PC
// runtime should memcpy outputs
// test should have a reference function to compare the output against

__global__ void runTest()
{
    if (allocateMemory(1024, 0) && allocateMemory(1024, 1024))
    {
        Runtime::loadBinary("saxpy.cu");
        Runtime::setupKernelEntryPoint("main");
        Runtime::launchSimulation();
        Runtime::memcpy(1024, 2048, 1024);
         
    }

} 



int main(int argc, char** argv)
{
    util::HostReflection::create();

    runTest<<<1, 1, 0>>>();

    util::HostReflection::destroy();

}
