/*! \file   CoreSimThread.cpp
	\date   Saturday May 8, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The source file for the Core simulator of the thread class.
*/

#include <archaeopteryx/executive/interface/CoreSimThread.h>
namespace executive
{

__device__ CoreSimThread::CoreSimThread(CoreSimBlock* parentBlock, unsigned threadId)
: m_parentBlock(parentBlock), m_tId(threadId)
{
}

static Value getRegisterOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    const ir::RegisterOperand& reg = static_cast<const ir::RegisterOperand&>(operand); 

    return block->getRegister(threadId, reg.reg);
}

static Value getImmediateOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    const ir::ImmediateOperand& imm = static_cast<const ir::ImmediateOperand&>(operand); 

    return imm.uint;
}



static Value getPredicateOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    const ir::PredicateOperand& reg = static_cast<const ir::PredicateOperand&>(operand); 
	//FIX ME    
    
    Value value = block->getRegister(threadId, reg.reg);

    switch(reg.modifier)
    {
    case ir::PredicateOperand::StraightPredicate:
    {
        value = value;
        break;
    }
    // TODO
    }

    return value;
}

static Value getIndirectOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    const ir::IndirectOperand& indirect = static_cast<const ir::IndirectOperand&>(operand); 
    
    Value address = indirect.base + indirect.offset;

    //FIXMe    
    return address;
}


typedef Value (*GetOperandValuePointer)(const ir::Operand&, CoreSimBlock*, unsigned);

static GetOperandValuePointer getOperandFunctionTable[] = {
    getRegisterOperand,
    getImmediateOperand,
    getPredicateOperand,
    getIndirectOperand
};

static Value getOperand(const ir::Operand& operand, CoreSimBlock* block, unsigned threadId)
{
    GetOperandValuePointer function = getOperandFunctionTable[operand.mode];

    function(operand, block, threadId);
}

static PC executeAdd(ir::Instruction* instruction, PC pc)
{
    ir::Add* add = static_cast<ir::Add*>(instruction);

    Value a = getOperand(add->a, parentBlock, threadId);
    Value b = getOperand(add->b, parentBlock, threadId);

    Value d = a + b;

    setRegister(add->d, parentBlock, threadId, d);
}

static PC executeAnd(ir::Instruction* instruction, PC pc)
{
    ir::And* andd = static_cast<ir::And*>(instruction);

    Value a = getOperand(andd->a, parentBlock, threadId);
    Value b = getOperand(andd->b, parentBlock, threadId);

    Value d = a & b;

    setRegister(andd->d, parentBlock, threadId);
}

static PC executeAshr(ir::Instruction* instruction, PC pc)
{
    ir::Ashr* ashr = static_cast<ir::Ashr*>(instruction);

    SValue a = getOperand(ashr->a, parentBlock, threadId);
    Value b = getOperand(ashr->b, parentBlock, threadId);

    SValue d = a >> b;

    setRegister(ashr->d, parentBlock, threadId);
}

static PC executeAtom(ir::Instruction* instruction, PC pc)
{
    ir::Atom* atom = static_cast<ir::Atom*>(instruction);

    Value a = getOperand(atom->a, parentBlock, threadId);
    Value b = getOperand(atom->b, parentBlock, threadId);

    Value physical = parentBlock->translateVirtualToPhysical(a);

    //TO DO
    Value d = atomicAdd((void*)physical, b);

    setRegister(atom->d, parentBlock, threadId);
}

static PC executeBar(ir::Instruction* instruction, PC pc)
{
    ir::Bar* bar = static_cast<ir::Bar*>(instruction);

    Value a = getOperand(bar->a, parentBlock, threadId);
    Value b = getOperand(bar->b, parentBlock, threadId);

    parentBlock->barrier(threadId);

    setRegister(bar->d, parentBlock, threadId);
}

static PC executeBitcast(ir::Instruction* instruction, PC pc)
{
    ir::Bitcast* bitcast = static_cast<ir::Bitcast*>(instruction);

    Value a = getOperand(bitcast->a, parentBlock, threadId);

    d = a;

    setRegister(bitcast->d, parentBlock, threadId);
}

static PC executeBra(ir::Instruction* instruction, PC pc)
{
    ir::Bra* bra = static_cast<ir::Bra*>(instruction);

    Value a = getOperand(bra->a, parentBlock, threadId);

    //TO DO
    return a;
}

static PC executeFpext(ir::Instruction* instruction, PC pc)
{
    ir::Fpext* fpext = static_cast<ir::Fpext*>(instruction);

    Value a = getOperand(fpext->a, parentBlock, threadId);

    float temp = bitcast<float>(a); 
    double d = temp;

    setRegister(fpext->d, parentBlock, threadId);
}

static PC executeFptosi(ir::Instruction* instruction, PC pc)
{
    ir::Fptosi* fptosi = static_cast<ir::Fptosi*>(instruction);

    Value a = getOperand(fptosi->a, parentBlock, threadId);

    float temp = bitcast<float>(a);
    SValue d   = temp;

    setRegister(fptosi->d, parentBlock, threadId);
}

static PC executeFptoui(ir::Instruction* instruction, PC pc)
{
    ir::Fptoui* fptoui = static_cast<ir::Fptoui*>(instruction);

    Value a = getOperand(fptoui->a, parentBlock, threadId);

    float temp = bitcast<float>(a);
    Value d    = temp;

    setRegister(fptoui->d, parentBlock, threadId);
}

static PC executeFpTrunc(ir::Instruction* instruction, PC pc)
{
    ir::FpTrunc* fptrunc = static_cast<ir::FpTrunc*>(instruction);

    Value a = getOperand(fptrunc->a, parentBlock, threadId);

    double temp = bitcast<double>(a);
    float d     = temp;

    setRegister(fptrunc->d, parentBlock, threadId);
}

static PC executeLd(ir::Instruction* instruction, PC pc)
{
    ir::Ld* ld = static_cast<ir::Ld*>(instruction);

    Value a = getOperand(ld->a, parentBlock, threadId);
    Value b = getOperand(ld->b, parentBlock, threadId);

    Value physical = parentBlock->translateVirtualToPhysical(a);
    // handle data types other than 64-bit
    Value d = *(Value*)physical;

    setRegister(ld->d, parentBlock, threadId);
}

static PC executeLshr(ir::Instruction* instruction, PC pc)
{
    ir::Lshr* lshr = static_cast<ir::Lshr*>(instruction);

    Value a = getOperand(lshr->a, parentBlock, threadId);
    Value b = getOperand(lshr->b, parentBlock, threadId);

    Value d = a >> b;

    setRegister(lshr->d, parentBlock, threadId);
}

static PC executeMembar(ir::Instruction* instruction, PC pc)
{
    ir::Membar* membar = static_cast<ir::Membar*>(instruction);

    __threadfence_block();
}

static PC executeMul(ir::Instruction* instruction, PC pc)
{
    ir::Mul* mul = static_cast<ir::Mul*>(instruction);

    Value a = getOperand(mul->a, parentBlock, threadId);
    Value b = getOperand(mul->b, parentBlock, threadId);

    Value d = a * b;

    setRegister(mul->d, parentBlock, threadId);
}

static PC executeOr(ir::Instruction* instruction, PC pc)
{
    ir::Or* orr = static_cast<ir::Or*>(instruction);

    Value a = getOperand(orr->a, parentBlock, threadId);
    Value b = getOperand(orr->b, parentBlock, threadId);

    Value d = a | b;

    setRegister(orr->d, parentBlock, threadId);
}
static PC executeRet(ir::Instruction* instruction, PC pc)
{
    ir::Ret* ret = static_cast<ir::Ret*>(instruction);

    parentBlock->returned(threadId); 
}

static PC executeSetP(ir::Instruction* instruction, PC pc)
{
    ir::SetP* setp = static_cast<ir::Setp*>(instruction);

    Value a = getOperand(setp->a, parentBlock, threadId);
    Value b = getOperand(setp->b, parentBlock, threadId);

    //TO DO
    Value d = a > b ? 1 : 0 ;
    setRegister(setp->d, parentBlock, threadId);
}

static PC executeSext(ir::Instruction* instruction, PC pc)
{
    ir::Sext* sext = static_cast<ir::Sext*>(instruction);

    Value a = getOperand(sext->a, parentBlock, threadId);

    int temp = bitcast<int>(a);
    SValue d = temp;

    setRegister(sext->d, parentBlock, threadId);
}

static PC executeSdiv(ir::Instruction* instruction, PC pc)
{
    ir::Sdiv* sdiv = static_cast<ir::Sdiv*>(instruction);

    Value a = getOperand(sdiv->a, parentBlock, threadId);
    Value b = getOperand(sdiv->b, parentBlock, threadId);

    //TO DO
    SValue d = (SValue) a / (SValue) b;
    setRegister(sdiv->d, parentBlock, threadId);
}

static PC executeShl(ir::Instruction* instruction, PC pc)
{
    ir::Shl* shl = static_cast<ir::Shl*>(instruction);

    Value a = getOperand(shl->a, parentBlock, threadId);
    Value b = getOperand(shl->b, parentBlock, threadId);

    Value d = a << b;

    setRegister(shl->d, parentBlock, threadId);
}

static PC executeSitofp(ir::Instruction* instruction, PC pc)
{
    ir::Sitofp* sitofp = static_cast<ir::Sitofp*>(instruction);

    Value a = getOperand(sitofp->a, parentBlock, threadId);

    //TO DO
    float d = (SValue)a;

    setRegister(sitofp->d, parentBlock, threadId);
}

static PC executeSrem(ir::Instruction* instruction, PC pc)
{
    ir::Srem* srem = static_cast<ir::Srem*>(instruction);

    SValue a = getOperand(srem->a, parentBlock, threadId);
    SValue b = getOperand(srem->b, parentBlock, threadId);

    SValue d = a % b;

    setRegister(srem->d, parentBlock, threadId);
}

static PC executeSt(ir::Instruction* instruction, PC pc)
{
    ir::St* st = static_cast<ir::St*>(instruction);

    Value a = getOperand(st->a, parentBlock, threadId);
    Value b = getOperand(st->b, parentBlock, threadId);

    //TO DO
    Address d = atAddress(a);
    
    *((Value*)d) = b;
}

static PC executeSub(ir::Instruction* instruction, PC pc)
{
    ir::Sub* sub = static_cast<ir::Sub*>(instruction);

    Value a = getOperand(sub->a, parentBlock, threadId);
    Value b = getOperand(sub->b, parentBlock, threadId);

    Value d = a - b;

    setRegister(sub->d, parentBlock, threadId);
}

static PC executeTrunc(ir::Instruction* instruction, PC pc)
{
    ir::Trunc* trunc = static_cast<ir::Trunc*>(instruction);

    Value a = getOperand(trunc->a, parentBlock, threadId);

    //TO DO
    Value d = uint32 (a & 0x00000000FFFFFFFFULL); 

    setRegister(trunc->d, parentBlock, threadId);
}

static PC executeUdiv(ir::Instruction* instruction, PC pc)
{
    ir::Udiv* udiv = static_cast<ir::Udiv*>(instruction);

    Value a = getOperand(udiv->a, parentBlock, threadId);
    Value b = getOperand(udiv->b, parentBlock, threadId);

    //TO DO
    Value d = a / b;

    setRegister(udiv->d, parentBlock, threadId);
}

static PC executeUitofp(ir::Instruction* instruction, PC pc)
{
    ir::Uitofp* uitofp = static_cast<ir::Uitofp*>(instruction);

    Value a = getOperand(uitofp->a, parentBlock, threadId);

    //TO DO
    float d = a;

    setRegister(uitofp->d, parentBlock, threadId);
}

static PC executeUrem(ir::Instruction* instruction, PC pc)
{
    ir::Urem* urem = static_cast<ir::Urem*>(instruction);

    Value a = getOperand(urem->a, parentBlock, threadId);
    Value b = getOperand(urem->b, parentBlock, threadId);

    //TO DO
    Value d = a % b;
    setRegister(urem->d, parentBlock, threadId);
}

static PC executeXor(ir::Instruction* instruction, PC pc)
{
    ir::Xor* xorr = static_cast<ir::Xor*>(instruction);

    Value a = getOperand(xorr->a, parentBlock, threadId);
    Value b = getOperand(xorr->b, parentBlock, threadId);

    Value d = a ^ b;

    setRegister(xorr->d, parentBlock, threadId);
}

static PC executeZext(ir::Instruction* instruction, PC pc)
{
    ir::Zext* zext = static_cast<ir::Zext*>(instruction);

    Value a = getOperand(zext->a, parentBlock, threadId);
    Value b = getOperand(zext->b, parentBlock, threadId);

    //TO DO
    uint64 d = (uint32)a;
    setRegister(zext->d, parentBlock, threadId);
}

static PC executeInvalidOpcode(ir::Instruction* instruction, PC pc)
{
    // TODO add this
    ir::InvalidOpcode* execinval = static_cast<ir::InvalidOpcode*>(instruction);

    //TODO: Add exceptions
}




typedef PC (*JumpTablePointer)(ir::Instruction*, PC);

static __device__ JumpTablePointer decodeTable[] = 
{
    executeAdd,
    executeAnd,
    executeAshr,
    executeAtom,
    executeBar,
    executeBitcast,
    executeBra,
    executeFpext,
    executeFptosi,
    executeFptoui,
    executeFptrunc,
    executeLd,
    executeshr,
    executeMembar,
    executeMul,
    executeOr,
    executeRet,
    executeSetP,
    executeSext,
    executeSdiv,
    executeShl,
    executeSitofp,
    executeSrem,
    executeSt,
    executeSub,
    executeTrunc,
    executeUdiv,
    executeUitofp,
    executeUrem,
    executeXor,
    executeZext,
    executeInvalidOpcode
};

__device__ PC executeInstruction(ir::Instruction* instruction, PC pc)
{
    JumpTablePointer decodedInstruction = decodeTable[instruction->opcode];

    decodedInstruction(instruction, pc);
}

}

