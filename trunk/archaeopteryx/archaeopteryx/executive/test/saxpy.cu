/*! \file   saxpy.cu
    \date   Saturday Feburary 26, 2011
    \author Gregory Diamos and Sudnya Diamos
        <gregory.diamos@gatech.edu, mailsudnya@gmail.com>
    \brief  A test for VIR and the simulator core.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Instruction.h>

template<typename T>
__device__ T getParameter(void* parameter, unsigned int byte = 0)
{
    return *(T*)((char*)parameter + byte);
}

/*
saxpy(int* y, int* x, int a)


    Begin:
        bitcast r11, "parameter_base";   // get address
        ld      r0, [r11]; // r0 is base of y
        ld      r1, [r11+8]; // r1 is base of x
        ld      r2, [r11+16]; // r2 is alpha

        bitcast r3,  "global_thread_id";
        zext    r12, r3;
        mul,    r4,  r12, 4;
        
        add     r5, r4, r0; // r5 is y[i]
        add     r6, r4, r1; // r6 is x[i]
        
        ld      r7, [r5];
        ld      r8, [r6];
        
        mul     r9,  r8, r2;
        add     r10, r7, r9;
        
        st      [r5], r10;
        
*/
__device__ void createSaxpy(void* parameters)
{
    ir::InstructionContainer* vir = 
        getParameter<ir::InstructionContainer*>(parameter);

    {
        ir::Bitcast& bitcast = vir[0].asBitcast; 
        
        bitcast.d.mode = ir::Operand::Register;
        bitcast.d.type = ir::Operand::i64;
        bitcast.d.reg  = 11;
        
        bitcast.a.mode = ir::Operand::Register;
        bitcast.a.type = ir::Operand::i64;
        bitcast.a.reg  = 32;
    }
    
    {
        ir::Ld& load = vir[1].asLoad; 
    
        load.d.mode = ir::Operand::Register;
        load.d.type = ir::Operand::i64;
        load.d.reg  = 0;
        
        load.a.mode   = ir::Operand::Indirect;
        load.a.type   = ir::Operand::i64;
        load.a.reg    = 11;
        load.a.offset = 0;
    }
    
    {
        ir::Ld& load = vir[2].asLoad; 
    
        load.d.mode   = ir::Operand::Register;
        load.d.type   = ir::Operand::i64;
        load.d.reg    = 1;
        
        load.a.mode   = ir::Operand::Indirect;
        load.a.type   = ir::Operand::i64;
        load.a.reg    = 11;
        load.a.offset = 8;
    }
    
    {
        ir::Ld& load = vir[3].asLoad; 
    
        load.d.mode   = ir::Operand::Register;
        load.d.type   = ir::Operand::i32;
        load.d.reg    = 2;
        
        load.a.mode   = ir::Operand::Indirect;
        load.a.type   = ir::Operand::i64;
        load.a.reg    = 11;
        load.a.offset = 16;
    }
    
    {
        ir::Bitcast& bitcast = vir[4].asBitcast; 
        
        bitcast.d.mode = ir::Operand::Register;
        bitcast.d.type = ir::Operand::i32;
        bitcast.d.reg  = 3;
        
        bitcast.a.mode = ir::Operand::Register;
        bitcast.a.type = ir::Operand::i32;
        bitcast.a.reg  = 33;
    }
    
    {
        ir::Zext& zext = vir[5].asZext; 
        
        zext.d.mode = ir::Operand::Register;
        zext.d.type = ir::Operand::i64;
        zext.d.reg  = 12;
        
        zext.a.mode = ir::Operand::Register;
        zext.a.type = ir::Operand::i32;
        zext.a.reg  = 3;
    }

    {
        ir::Mul& multiply = vir[6].asMul; 
        
        multiply.d.mode = ir::Operand::Register;
        multiply.d.type = ir::Operand::i64;
        multiply.d.reg  = 4;
        
        multiply.a.mode = ir::Operand::Register;
        multiply.a.type = ir::Operand::i64;
        multiply.a.reg  = 12;

        multiply.b.mode = ir::Operand::Immediate;
        multiply.b.type = ir::Operand::i64;
        multiply.b.uint = 4;
    }

    {
        ir::Add& add = vir[7].asAdd; 
        
        add.d.mode = ir::Operand::Register;
        add.d.type = ir::Operand::i64;
        add.d.reg  = 5;
        
        add.a.mode = ir::Operand::Register;
        add.a.type = ir::Operand::i64;
        add.a.reg  = 4;

        add.b.mode = ir::Operand::Register;
        add.b.type = ir::Operand::i64;
        add.b.reg  = 0;
    }

    {
        ir::Add& add = vir[8].asAdd; 
        
        add.d.mode = ir::Operand::Register;
        add.d.type = ir::Operand::i64;
        add.d.reg  = 6;
        
        add.a.mode = ir::Operand::Register;
        add.a.type = ir::Operand::i64;
        add.a.reg  = 4;

        add.b.mode = ir::Operand::Register;
        add.b.type = ir::Operand::i64;
        add.b.reg  = 1;
    }
    
    {
        ir::Ld& load = vir[9].asLoad; 
    
        load.d.mode   = ir::Operand::Register;
        load.d.type   = ir::Operand::i32;
        load.d.reg    = 7;
        
        load.a.mode   = ir::Operand::Indirect;
        load.a.type   = ir::Operand::i64;
        load.a.reg    = 5;
        load.a.offset = 0;
    }
    
    {
        ir::Ld& load = vir[10].asLoad; 
    
        load.d.mode   = ir::Operand::Register;
        load.d.type   = ir::Operand::i32;
        load.d.reg    = 8;
        
        load.a.mode   = ir::Operand::Indirect;
        load.a.type   = ir::Operand::i64;
        load.a.reg    = 6;
        load.a.offset = 0;
    }
    
    {
        ir::Mul& multiply = vir[11].asMul; 
        
        multiply.d.mode = ir::Operand::Register;
        multiply.d.type = ir::Operand::i32;
        multiply.d.reg  = 9;
        
        multiply.a.mode = ir::Operand::Register;
        multiply.a.type = ir::Operand::i32;
        multiply.a.reg  = 8;

        multiply.b.mode = ir::Operand::Register;
        multiply.b.type = ir::Operand::i32;
        multiply.b.reg  = 2;
    }

    {
        ir::Add& add = vir[12].asAdd; 
        
        add.d.mode = ir::Operand::Register;
        add.d.type = ir::Operand::i32;
        add.d.reg  = 10;
        
        add.a.mode = ir::Operand::Register;
        add.a.type = ir::Operand::i32;
        add.a.reg  = 7;

        add.b.mode = ir::Operand::Register;
        add.b.type = ir::Operand::i32;
        add.b.reg  = 9;
    }

    {
        ir::St& store = vir[13].asStore; 
    
        store.d.mode   = ir::Operand::Indirect;
        store.d.type   = ir::Operand::i64;
        store.d.reg    = 5;
        store.d.offset = 0;
        
        store.a.mode   = ir::Operand::Register;
        store.a.type   = ir::Operand::i32;
        store.a.reg    = 10;
    }
}

typedef long long unsigned int uint64;
typedef long long unsigned int Register;
typedef Register* RegisterFile;

class SimulatorState 
{
    public:
        uint64 globalMemoryWindowHi;
        uint64 globalMemoryWindowLow;
        void*  globalMemoryWindow;
        uint64 baseProgramCounter;
        RegisterFile registerFile;
        ir::InstructionContainer* instructionMemory;
    public:
        __device__ SimulatorState(uint64 globalMemoryWindowHi, 
            uint64 globalMemoryWindowLow, void* globalMemoryWindow,
            uint64 baseProgramCounter, RegisterFile registerFile,
            ir::InstructionContainer* instructionMemory);
};
////////////////////////////////////////////////////////////////////////////////
// setupSimulatorState
////////////////////////////////////////////////////////////////////////////////
__device__ void setupSimulatorState(void* parameters)
{
    SimulatorState* state = getParameter<SimulatorState*>(parameters, 0);
    ir::InstructionContainter* instructionMemory = 
        getParameter<ir::InstructionContainer*>(parameters, sizeof(SimulatorState*));

    RegisterFile registerFile = (RegisterFile)std::malloc(sizeof(Register)*64);
    void* globalMemoryWindow = std::malloc(0x84);

    new(state) SimulatorState(0x84, 0x0, globalMemoryWindow,
        0, registerFile, instructionMemory);
}

////////////////////////////////////////////////////////////////////////////////
// runSimulation
////////////////////////////////////////////////////////////////////////////////
__device__ void runSimulation(void* parameters)
{
    SimulatorState* state = getParameter<SimulatorState*>(parameters, 0);
    RegisterFile* registerFile = state->registerFile;
    uint64 pc = state->baseProgramCounter;
    bool running = true;
    while(running)
    {
        ir::InstructionContainer instruction = state->instructionMemory[pc];
        
        ir::Instruction& baseInstruction = instruction.asInstruction;
        
        switch(baseInstruction.opcode)
        {
            case ir::Instruction::Add:
                {
                    ir::Add& add = instruction.asAdd;
                    
                    ir::RegisterType aId = add.a.asRegister.reg;
                    ir::RegisterType bId = add.b.asRegister.reg;
                    ir::RegisterType dId = add.d.asRegister.reg;

                    Register a = registerFile[aId];
                    Register b = registerFile[bId];
                    Register d = 0;
                    
                    switch(add.a.asIndirect.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                d = (char)a + (char)b;
                                break;
                            }
                        case ir::i16:
                            {
                                d = (short)a + (short)b;
                                break;
                            }
                        case ir::i32:
                            {
                                d = (int)a + (int)b;
                                break;
                            }
                        case ir::f32:
                            {
                                d = float_cast(a) + float_cast(b);
                                break;
                            }
                        case ir::i64:
                            {
                                d = (long long int)a + (long long int)b;
                                break;
                            }
                        case ir::f64:
                            {
                                d = double_cast(a) + double_cast(b);
                                break;
                            }
                        default: break;
                    }
                    ++pc;
                    break;
                }
            case ir::Instruction::Bitcast:
                {
                    ir::Bitcast& bitcast = instruction.asBitcast;
                    
                    ir::RegisterType aId = bitcast.a.asRegister.reg;
                    ir::RegisterType dId = bitcast.d.asRegister.reg;
                    
                    registerFile[dId] = registerFile[aId];
                    ++pc;
                    break;
                }
            case ir::Instruction::Ld:
                {
                    ir:::Ld& load = instruction.asLd;
                    
                    ir::RegisterType dId = load.d.asRegister.reg;
                    ir::RegisterType aId = load.a.asIndirect.reg;
                    int offset = load.a.asIndirect.offset;
                    uint64 vaddress = registerFile[aId];
                    vaddress += offset;
                    uint64 base = (uint64)state->globalMemoryWindow;
                    uint64 address = vaddress - state->globalMemoryWindowLow
                        + base;
                    
                    Register value = 0;
                    
                    switch(load.a.asIndirect.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                value = *((char*)address);
                                break;
                            }
                        case ir::i16:
                            {
                                value = *((short*)address);
                                break;
                            }
                        case ir::i32: /* fall through */
                        case ir::f32:
                            {
                                value = *((int*)address);
                                break;
                            }
                        case ir::i64:
                        case ir::f64:
                            {
                                value = *((long long int*)address);
                                break;
                            }
                        default: break;
                    }
                    
                    registerFile[dId] = value;
                    ++pc;
                    break;
                }
            case ir::Instruction::Mul:
                {
                    ir::Mul& mul = instruction.asMul;
                    
                    ir::RegisterType dId = mul.d.asRegister.reg;

                    Register a = 0;
                    Register b = 0;
                    Register d = 0;

                    if(bitcast.a.mode == ir::OperandBase::Register)
                    {
                        ir::RegisterType aId = mul.a.asRegister.reg;
                        
                        a = registerFile[aId];
                    }
                    else
                    {
                        switch(mul.a.asImmediate.type)
                        {
                            case ir::i1:  /* fall through */
                            case ir::i8:  /* fall through */
                            case ir::i16: /* fall through */
                            case ir::i32: /* fall through */
                            case ir::i64:
                                {
                                    a = mul.a.asImmediate.uint;
                                    break;
                                }
                            case ir::f32:
                                {
                                    a = float_cast(
                                        (float)mul.a.asImmediate.fp);
                                    break;
                                }
                            case ir::f64:
                                {
                                    a = mul.a.asImmediate.fp;
                                    break;
                                }
                            default: break;                        
                        }
                    }

                    if(bitcast.a.mode == ir::OperandBase::Register)
                    {
                        ir::RegisterType bId = mul.b.asRegister.reg;
                        
                        b = registerFile[bId];
                    }
                    else
                    {
                        switch(mul.b.asImmediate.type)
                        {
                            case ir::i1:  /* fall through */
                            case ir::i8:  /* fall through */
                            case ir::i16: /* fall through */
                            case ir::i32: /* fall through */
                            case ir::i64:
                                {
                                    b = mul.b.asImmediate.uint;
                                    break;
                                }
                            case ir::f32:
                                {
                                    b = float_cast(
                                        (float)mul.b.asImmediate.fp);
                                    break;
                                }
                            case ir::f64:
                                {
                                    b = mul.b.asImmediate.fp;
                                    break;
                                }
                            default: break;                        
                        }
                    }
                    
                    switch(mul.a.asIndirect.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                d = (char)a * (char)b;
                                break;
                            }
                        case ir::i16:
                            {
                                d = (short)a * (short)b;
                                break;
                            }
                        case ir::i32:
                            {
                                d = (int)a * (int)b;
                                break;
                            }
                        case ir::f32:
                            {
                                d = float_cast(a) * float_cast(b);
                                break;
                            }
                        case ir::i64:
                            {
                                d = (long long int)a * (long long int)b;
                                break;
                            }
                        case ir::f64:
                            {
                                d = double_cast(a) * double_cast(b);
                                break;
                            }
                        default: break;
                    }
                    ++pc;
                    break;
                }
            case ir::Instruction::Ret:
                {
                    running = false;
                    break;
                }
            case ir::Instruction::St:
                {
                    ir:::St& store = instruction.asSt;
                    
                    ir::RegisterType aId = store.a.asRegister.reg;
                    ir::RegisterType dId = store.d.asIndirect.reg;
                    int offset = store.d.asIndirect.offset;
                    uint64 vaddress = registerFile[dId];
                    vaddress += offset;
                    uint64 base = (uint64)state->globalMemoryWindow;
                    uint64 address = vaddress - state->globalMemoryWindowLow
                        + base;
                    
                    Register value = registerFile[aId];

                    switch(load.d.asIndirect.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                *((char*)address) = value;
                                break;
                            }
                        case ir::i16:
                            {
                                *((short*)address) = value;
                                break;
                            }
                        case ir::i32: /* fall through */
                        case ir::f32:
                            {
                                *((int*)address) = value;
                                break;
                            }
                        case ir::i64:
                        case ir::f64:
                            {
                                *((long long int*)address) = value;
                                break;
                            }
                        default: break;
                    }
                    ++pc;
                    break;
                }
            case ir::Instruction::Zext:
                {
                    ir:::Zext& zext = instruction.asZext;
                    
                    ir::RegisterType dId = load.d.asRegister.reg;
                    ir::RegisterType aId = load.a.asRegister.reg;
                    
                    Register a = registerFile[aId];
                    Register d = 0;
                    
                    switch(load.a.asRegister.type)
                    {
                        case ir::i1: /* fall through */
                        case ir::i8:
                            {
                                d = (unsigned char)a;
                                break;
                            }
                        case ir::i16:
                            {
                                d = (unsigned short)a;
                                break;
                            }
                        case ir::i32: /* fall through */
                        case ir::f32:
                            {
                                d = (unsigned int)a;
                                break;
                            }
                        case ir::i64:
                        case ir::f64:
                            {
                                d = a;
                                break;
                            }
                        default: break;
                    }
                    
                    registerFile[dId] = d;
                    ++pc;
                    break;
                }
            default:
                {
                    ++pc;
                    break;
                }
        }
    }
}

#include <cstdlib>

__global__ void system()
{
    void* instructionMemory = std::malloc(14*sizeof(ir::InstructionContainer));
    // 1) call createSaxypy()
    system_call("createSaxpy", instructionMemory);
    //    __bar()
    // 2) call setupSimulatorState()
    SimulatorState* state = (SimulatorState*)std::malloc(sizeof(SimulatorState));
    system_call("setupSimulatorState", state, instructionMemory);
    //    __bar()
    // 3) call runSimulation()
    system_call("runSimulation", state);
    //    __bar()
    std::free(instructionMemory);
    std::free(state);
}

