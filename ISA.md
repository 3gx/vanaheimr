# Scope #

The purpose of the IR is to provide a low level virtual instruction set that is simple in its ability to capture the core features of programs and extendable such that new features or intrinsics can augment the core ISA for experimentation.

# The Core ISA #

## Data Types ##

Most of the LLVM type system is adopted here with the explicit limit to common word sizes and notion of a predicate register.

  * i1  - one bit predicate
  * i8  - eight bit int
  * i16 - sixteen bit int
  * i32 - thirty-two bit int
  * i64 - sixty-four bit int
  * f32 - IEEE754 float
  * f64 - IEEE754 double

# Aggregate Types #

Core data types may be aggregated together to form structures, arrays,
and pointers.

## Values ##

The Vanaheimr ISA uses a strict SSA form for all instructions.  Values
are uniquely defined once, either by an instruction or as a global that
dominates the program entry point.  PHI join nodes are explicit in the
dataflow representation of the program.

## Predicates ##

All instructions support an optional predicate operand, which controls
the conditional execution of the instruction.  PSI join nodes are made
explicit in the dataflow representation of a program.

## Instructions ##

### Arithmetic and Logical ###

  * add  - two operand add operation
  * sub  - two operand subtract operation
  * mul  - two operand multiply operation
  * udiv - two operand unsigned divide
  * sdiv - two operand signed divide
  * urem - two operand unsigned remainder
  * srem - two operand signed remainder

  * xor - two operand bitwise xor
  * or  - two operand bitwise or
  * and - two operand bitwise and

  * lsl - logical shift left
  * asr - sign extension shift right
  * lsr - logical shift right

  * setp - Compare two operands, set a predicate

### Control Flow ###

  * bra  - branch to a label/register based on a condition, indirect branching is also supported
  * call - branch to a label/register based on a condition, takes a list of values as parameters, allows returning via ret.

### Data Manipulation ###

  * bitcast - Move one register into another of the same size, type is ignored
  * fpext   - floating point conversion from f32 to f64
  * fptosi  - floating point conversion to signed integer
  * fptoui  - floating point conversion to unsigned integer
  * fptrunc - floating point conversion from f64 to f32
  * sext    - convert to a larger integer size using sign extension
  * sitofp  - signed integer conversion to floating point
  * trunc   - integer conversion to a smaller size
  * uitofp  - unsigned integer conversion to floating point
  * zext    - zero extend and integer to a larger size

### Memory Interaction ###

  * ld   - load instruction
  * st   - store instruction

### Synchronization ###

  * bar    - barrier at various levels of granularity (each corresponding to a level in the thread hierarchy)
  * fence  - wait until all previous memory requests are visible at the specified level in the thread hierarchy

### Task Creation ###
  * launch - similar to call, but creates a new BSP kernel launch and stores a handle to the launch in a register.

## Intrinsics ##

intrinsic operations are opaque, they have inputs/outputs, variable latency, and do not affect control flow

  * atom{.operation} - perform a series of operations on memory atomically

## Modifiers ##

### Comparisons ###

Ordered operators apply to floating point or integer types (integers are
always ordered), Unordered and IsANumber/NotANumber apply only to floating
point types.

  * oe  - OrderedEqual            - `a == b`
  * one - OrderedNotEqual         - `a != b`
  * olt - OrderedLessThan         - `a <  b`
  * ole - OrderedLessOrEqual      - `a <= b`
  * ogt - OrderedGreater          - `a >  b`
  * oge - OrderedGreaterOrEqual   - `a >= b`
  * ue  - UnorderedEqual          - `a == b || isnan(a) || isnan(b)`
  * une - UnorderedNotEqual       - `a != b || isnan(a) || isnan(b)`
  * ult - UnorderedLessThan       - `a <  b || isnan(a) || isnan(b)`
  * ule - UnorderedLessOrEqual    - `a <= b || isnan(a) || isnan(b)`
  * ugt - UnorderedGreaterThan    - `a >  b || isnan(a) || isnan(b)`
  * uge - UnorderedGreaterOrEqual - `a >= b || isnan(a) || isnan(b)`
  * num - IsANumber               - `!isnan(a) && !isnan(b)`
  * nan - NotANumber              - `isnan(a) || isnan(b)`

### Atomic Operations ###

These take at least two operands:
  * a - the address being modified
  * b - the first input
  * c - the second input (optional)
  * d - the result (output)

Ops:
  * And  - atomic { d = [a](a.md); [a](a.md) = b & c; }
  * Or   - atomic { d = [a](a.md); [a](a.md) = b | c; }
  * Xor  - atomic { d = [a](a.md); [a](a.md) = b ^ c; }
  * Cas  - atomic { d = [a](a.md); [a](a.md) = d == b ? c : d; }
  * Exch - atomic { d = [a](a.md); [a](a.md) = b; }
  * Add  - atomic { d = [a](a.md); [a](a.md) = b + c; }
  * Inc  - atomic { d = [a](a.md); [a](a.md) = (d >= b) ? 0 : d + 1; }
  * Dec  - atomic { d = [a](a.md); [a](a.md) = ((d == 0) || (d > b)) ? b : d - 1; }
  * Min  - atomic { d = [a](a.md); [a](a.md) = min(b, c); }
  * Max  - atomic { d = [a](a.md); [a](a.md) = max(b, c); }

# Memory Model #

The memory model of VIR is tightly coupled with the ISA to create an association between computation and data that can be reasoned about by a compiler or runtime system.

Automatic variables are defined with a scoping level. A scoping level indicates which level of the thread hierarchy the memory is associated with. Level 0 is per threads, level 1 is per thread group, and so on.

Dynamic memory allocations are also defined with a scoping level.
The default level is 0, which makes allocations local to the thread
that allocated them, but still globally accessible.  Allocations at other levels may be used for more efficient data sharing among threads.

# ABI #
**TBD** although the low level details regarding calling conventions and memory layout should
not be exposed here.

# Supported Extensions #

## Intrinsics ##

Intrinsics take the form of function calls with a list of basic types as parameters.  Intrinsics must express dependencies on either the memory system or a set of operands to capture data dependencies.  Intrinsics cannot change the control flow of the program until we have a good set of abstractions to deal with it.