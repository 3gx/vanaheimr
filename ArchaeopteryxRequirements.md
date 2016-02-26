# Vision #

![http://www.gdiamos.net/images/archaeopteryx-overview.png](http://www.gdiamos.net/images/archaeopteryx-overview.png)

Enable **fast** simulation of future architectures on modern hardware.

# Goals #

  * Functionally simulate a many-core processor using GPUs.
  * Maintain a constant slowdown factor when simulating a new processor on current generation hardware.
  * Make all components modular, with well defined interfaces, especially with regards to the ISA.
  * Include ISA and runtime extensions for data distributions and locality.
  * Support lightweight and modular instrumentation or trace generation interface for interactions with timing models.

# Requirements #

  * Simulate complete CUDA applications.  As a starting point, simulate the PTX virtual machine (this may need revision).
  * A fast virtual machine IR and byte-code format.  Ideally, this should be loaded, and simulated in parallel.
  * Written completely in CUDA device code.  No host code at all for the simulator engine.

# Component List #
  * An IR and a byte-code format
  * Processor simulator
    * Kernel Instructions
    * Memory module
    * Core simulator
      * Instruction execution engine
  * Runtime
  * Event Handlers
  * Test framework

# Component Design #

This section covers the aforementioned components in more detail.

## IR and Byte-Code ##

The IR will be a set of CUDA classes that represent ISA abstractions, such as instructions or memory allocations.  The byte-code format will provide an efficient mechanism for storing and loading this IR to disk.  This format should allow files to be loaded in a data-parallel fashion that can achieve close to peak performance on the system performing the simulation.  It should also be possible to load the IR from the byte-code lazily.  It should not be necessary to load an entire file to discover the externally visible variables and functions.  It should also not be necessary to completely load a function to begin executing it.

## Processor Simulator ##

The processor simulator is the top level component.  It contains a collection of memory modules, core models, and kernel instructions that are fed to CTAs during the execution of the program. This dispatches work to the various core simulator instances. The physical interconnections between various components is not modeled here, instead all cores are assumed to be connected to each memory module. For now we adopt the convention in current architectures where cores are not directly connected to each other.

### Kernel Instructions ###

Kernel instructions will be stored in a pre-decoded IR form in the last level shared memory space of the machine performing the simulation.  The instructions for an entire program will be mapped into this memory space, but they should be loaded lazily from a byte-code file, one page at a time.  This will require some tricks to get working correctly with CUDA, but it should ensure that significantly less than the full binary needs to be loaded at a time into the memory of the machine performing the simulation.  Additional work will be needed to design an IR storage format that enables this behavior.

### Memory Modules ###

Memory modules represent relocatable pages of memory that are available to the currently executing application.  They will be associated with the current process only.  Modules will be associated with an allocation function, which controls their distribution on a system with multiple memory controllers and a multi-level tiled cache.

### Core Simulator ###

The core simulator is responsible for executing complete CTAs.  It will be given a starting instruction and a set of memory regions that it has access to.  There will also be a set of memory regions that are shared across all CTAs, but are assumed to support more efficient access from this CTA.

Instructions will be loaded lazily.  They should be grouped into pages and only allocated and read from the byte-code file when the first thread executes them.  The Core simulator should cache valid regions of instructions and atomically update a shared data structure containing the loaded instruction pages.

#### Instruction Execution Engine ####

This component is responsible for executing single instructions at a time.  It should maintain per-thread local state (register files, local stack pointers etc.) as well as interfaces to the per-CTA and per-processor shared state (mainly memory).

It will be used by the Core Simulator to execute individual instructions.  The idea is that the Core Simulator will get the next instruction to execute and call the Execution Engine to execute it.  This will be equivalent to a switch statement over the opcode followed by instruction specific operations (although performance considerations may mandate a different implementation).

## Runtime ##

The runtime component will be a low level interface that allows the configuration and execution of kernels on the simulator, allowing the simulator to be run as a back-end to a variety of front-ends languages, such as CUDA or OpenCL via [Ocelot](http://code.google.com/p/gpuocelot/).

Functionality will be provided for the following basic operations:
  * Lazily loading a byte-code file for execution.
  * Allocating and configuring memory.
  * Setup the parameters of the next kernel and perform the launch.
  * Registering simulation event handlers.

No other advanced functions such as concurrent simulation of programs or asynchronous operations will be supported at this level.

## Event Handlers ##

Event handlers are user-defined functions that implement an interface that receives callbacks when pre-defined events occur during simulation.  In the most basic case, event handlers are attached to instruction execution engines, and a callback is triggered after each instruction is executed.  The interface should be designed such that many event handlers can be registered at a time, and many instances of each event handler can be running in parallel, attached to different core simulators, memory modules, or instruction execution engines.

## Test framework ##

The archaeopteryx testing framework is designed to go beyond simple unit and module level directed testing to automated test-space exploration, distributed testing, and algorithm assisted bug diagnosis.  Tests are specified over ranges of valid inputs. Feature spaces are explicitly defined along with associated coverages.  Test dependencies are included so that changes to the code base do not necessitate re-execution of tests over unrelated features.

### Unit tests ###

Unit tests exercise the behavior of individual software components in isolation.

#### Feature coverage ####

Features describe some aspect of the system being tested.  Features may include the correct operation of floating point, or memory instructions.  They may also be composed to created aggregates, such as the correct operations of loops.  Tests should list the complete set of features that they are allowed to exercise.

#### Input parameters ####

Input parameters may be given explicitly in terms of exact values, as a bounded range of possible values, or as a random function that generates a set of valid inputs given an initial seed.

#### Output parameters ####

Output parameters are the expected results of a test.  They can be specified explicitly in the case of exact input values, or as a reference implementation that can be fed arbitrary inputs to produce golden outputs.

### Test space exploration ###

The test space is a multi-dimensional space with one dimension for each feature.  The combination of a single test and a single set of inputs define a point, line, plane, or higher dimension hyperplane depending on the features covered by the test.  Tests that are defined over a range of inputs will define a bounded region in the dimensions of each feature that it covers, and a projection into the independent spaces.

Note that this allows tests to have overlapping coverage.  Exploring the test space may be accomplished by running the tests with the greatest coverage until the maximum area is covered.  For tests whose feature space completely contains other tests, the contained tests may be run to isolate bugs by excluding features which do not cause the bug.

### Random program generation ###

One definition for the correctness of this simulator is that it should be able to
execute all possible applications that can be expressed in the ISA.  The test space
for this is obviously too large to explore manually.  It would be useful to be able
generate programs automatically with certain statistical properties and deterministic results.

### Test Dependencies ###

In order to avoid invaliding an entire set of tests when a simulator component changes, tests should explicitly list dependencies on simulator components.

### The Test Database ###

Tests should be specified as a byte-code file, a set of memory inputs or an input generating function, a set of memory outputs or a reference generating function, a list of dependencies, and a list of features.  The database should maintain a list of all tests, all features, a versioned copy of the simulator.  New versions of the simulator should be added to the simulator along with a list of changes that can be used to determined dependent tests.  The database should keep a versioned history of feature set coverage and failure rates over each feature.

### Distributed Testing ###

Ideally, a set of nodes in a distributed system will be available for executing tests.  These nodes will access a shared database of tests that will describe the
entire space of features, the coverage of each space, and the set of available tests.  Each node will randomly walk the feature space using available test with the goal of maximizing coverage in a bounded amount of time.  A core assumption of this model is that covering the entire feature space is computationally infeasible, and that it is instead desirable to report the failure probability of each feature along with a confidence value. The nodes in the system should choose tests that are
expected to have the greatest impact on the detection of failures and the confidence
in failure rates.

When the system changes, the list of changes should be recorded, and only test results associated with changed components should be discarded.

# Infrastructure Requirements #

  * Access to GPU workstations or laptops
  * CUDA compiler

# Schedule #

## Milestone 1: April 15th 2011 ##
  * First SAXPY example running on 580 GTX
  * Completely using CUDA code (no CPU code)

## Milestone 2: June 15th ##
  * Most of the code in place
  * Ocelot sanity level regression test suit 100% passing

## Milestone 3: August 15th ##
  * Most CUDA applications running
  * Ocelot basic level regression test suit 100% passing

## Target: ISPASS 2012 (October 2011) ##
  * Self-Simulating
  * Simulate Parboil/Rodinia
  * Final paper draft
