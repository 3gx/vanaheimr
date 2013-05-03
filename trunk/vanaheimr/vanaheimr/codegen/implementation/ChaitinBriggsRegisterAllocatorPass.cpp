/*! \file   ChaitinBriggsRegisterAllocatorPass.cpp
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ChaitinBriggsRegisterAllocatorPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ChaitinBriggsRegisterAllocatorPass.h>

#include <vanaheimr/analysis/interface/InterferenceAnalysis.h>

#include <vanaheimr/machine/interface/MachineModel.h>
#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/VirtualRegister.h>

#include <vanaheimr/util/interface/SmallSet.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <algorithm>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace vanaheimr
{

namespace codegen
{

ChaitinBriggsRegisterAllocatorPass::ChaitinBriggsRegisterAllocatorPass()
: RegisterAllocator({"InterferenceAnalysis"},
	"ChaitinBriggsRegisterAllocatorPass")
{

}

typedef analysis::InterferenceAnalysis InterferenceAnalysis;
typedef util::LargeMap<unsigned int, unsigned int> RegisterMap;

static void color(RegisterAllocator::VirtualRegisterSet& spilled,
	RegisterMap& allocated, const ir::Function& function,
	const InterferenceAnalysis& interferences, unsigned int colors);

void ChaitinBriggsRegisterAllocatorPass::runOnFunction(Function& f)
{
	report("Running chaitin-briggs graph coloring register allocator on "
		<< f.name());
	
	auto interferenceAnalysis = static_cast<InterferenceAnalysis*>(
		getAnalysis("InterferenceAnalysis"));
	assert(interferenceAnalysis != nullptr);
	
	auto machineModel = compiler::Compiler::getSingleton()->getMachineModel();
	
	// attempt to color the interferences
	color(_spilled, _allocated, f, *interferenceAnalysis,
		machineModel->totalRegisterCount());
	
	// TODO: spill if allocation fails
	assertM(_spilled.empty(), "No support for spills yet.");
	
	// TODO: Map colors to registers
	
}

transforms::Pass* ChaitinBriggsRegisterAllocatorPass::clone() const
{
	return new ChaitinBriggsRegisterAllocatorPass;
}

RegisterAllocator::VirtualRegisterSet
	ChaitinBriggsRegisterAllocatorPass::getSpilledRegisters()
{
	return _spilled;
}

const machine::PhysicalRegister*
	ChaitinBriggsRegisterAllocatorPass::getPhysicalRegister(
	const ir::VirtualRegister& vr) const
{
	auto allocatedRegister = _allocated.find(vr.id);
	
	if(allocatedRegister == _allocated.end()) return nullptr;
	
	return _machine->getPhysicalRegister(allocatedRegister->second);
}

class RegisterInfo
{
public:
	RegisterInfo(const ir::VirtualRegister* r, unsigned int d = 0,
		unsigned int c = 0, unsigned int s = 0)
	: virtualRegister(r), nodeDegree(d), color(c), schedulingOrder(s)
	{
	
	}

public:
	const ir::VirtualRegister* virtualRegister;
	unsigned int               nodeDegree;
	unsigned int               color;
	unsigned int               schedulingOrder;

};

typedef std::vector<RegisterInfo> RegisterInfoVector;
	
typedef util::SmallSet<unsigned int> ColorSet;

static unsigned int computeColor(const RegisterInfo& reg,
	const RegisterInfoVector& registerInfo,
	const InterferenceAnalysis& interferences, unsigned int partitionSize)
{
	ColorSet usedColors;
	
	auto regInterferences =
		interferences.getInterferences(*reg.virtualRegister);

	for(auto interference : regInterferences)
	{
		assert(interference->id < registerInfo.size());
	
		const RegisterInfo& info = registerInfo[interference->id];

		if(info.schedulingOrder >= reg.schedulingOrder) continue;
		if(info.schedulingOrder >  partitionSize) continue;
	
		usedColors.insert(info.color);
	}

	unsigned int newColor = 0;
	
	for(auto used : usedColors)
	{
		if(newColor != used) break;
		
		++newColor;
	}
	
	return newColor;
}

static bool propagateColorsInParallel(RegisterInfoVector& registers,
	unsigned int iteration, const InterferenceAnalysis& interferences)
{
	report("  -------------------- Iteration "
		<< iteration << " ------------------");

	unsigned int partitionSize = 1 << iteration;

	RegisterInfoVector newRegisters;
	
	newRegisters.reserve(registers.size());
	bool changed = false;
	
	for(auto reg = registers.begin(); reg != registers.end(); ++reg)
	{
		unsigned int newColor = computeColor(*reg, registers,
			interferences, partitionSize);

		newRegisters.push_back(RegisterInfo(reg->virtualRegister,
			reg->nodeDegree, newColor, reg->schedulingOrder));

		changed |= reg->color != newColor;

		reportE(reg->color != newColor, "   vr" << reg->virtualRegister->id
			<< " (degree " << reg->nodeDegree
			<< ") | (color " << reg->color << ") -> (color " << newColor
			<< ")");
	}
	
	registers = std::move(newRegisters);
	
	return changed;
}

static void initializeSchedulingOrder(RegisterInfoVector& registerInfo)
{
	typedef std::pair<unsigned int, RegisterInfo*> DegreeAndInfoPair;
	typedef std::vector<DegreeAndInfoPair>         DegreeAndInfoVector;
	
	report(" Ranking registers by interference graph node degree");
	
	DegreeAndInfoVector degrees;
	
	degrees.reserve(registerInfo.size());
	
	for(auto info = registerInfo.begin(); info != registerInfo.end(); ++info)
	{
		degrees.push_back(std::make_pair(info->nodeDegree, &*info));
	}
	
	// Sort by node degree (parallel)
	std::sort(degrees.begin(), degrees.end(),
		std::greater<DegreeAndInfoPair>());

	for(auto degree = degrees.begin(); degree != degrees.end(); ++degree)
	{
		report("  vr" << degree->second->virtualRegister->id
			<< " (" << degree->first << ")");
		degree->second->schedulingOrder =
			std::distance(degrees.begin(), degree);
	}
}

static void color(RegisterAllocator::VirtualRegisterSet& spilled,
	RegisterMap& allocated, const ir::Function& function,
	const InterferenceAnalysis& interferences, unsigned int colors)
{
	// Create a map from node degree to virtual register
	RegisterInfoVector registers;
	
	registers.reserve(function.register_size());
	
	for(auto reg = function.register_begin();
		reg != function.register_end(); ++reg)
	{
		registers.push_back(RegisterInfo(&*reg,
			interferences.getInterferences(*reg).size()));
	}
	
	// Initialize scheduling order for each node
	initializeSchedulingOrder(registers);
	
	// Propagate colors until converged
	report(" Propating colors until converged.");
	
	unsigned int iteration = 0;
	bool changed = true;
	
	while(changed)
	{
		changed = propagateColorsInParallel(registers,
			iteration++, interferences);
		
		// Check iteration count
		assertM(iteration <= colors, "Too many iterations: " << iteration);
	}
	
}

}

}


