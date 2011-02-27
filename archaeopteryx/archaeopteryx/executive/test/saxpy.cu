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
		bitcast r11, "parameter_base";   // get addres
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




