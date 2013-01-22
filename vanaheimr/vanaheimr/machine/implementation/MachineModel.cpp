/*! \file   MachineModel.h
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MachineModel class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/MachineModel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace machine
{

MachineModel::MachineModel(const std::string& n)
: name(n)
{

}

const PhysicalRegister* MachineModel::getPhysicalRegister(RegisterId id) const
{
	assertM(false, "Not implemented.");
	
	return nullptr;
}

unsigned int MachineModel::totalRegisterCount() const
{
	assertM(false, "Not implemented.");

	return 0;
}

void MachineModel::configure(const StringVector& )
{
	// TODO
}

std::string makeRegisterName(const RegisterFile& file, unsigned int id)
{
	std::stringstream stream;
	
	stream << file.name() << id;
	
	return stream.str();
}

void MachineModel::addRegisterFile(const std::string& name,
	unsigned int registers)
{
	auto file = _registerFiles.insert(std::make_pair(name,
		RegisterFile(this, name))).first;

	for(unsigned int i = 0; i < registers; ++i)
	{
		file->second.registers.push_back(PhysicalRegister(&file->second, i,
			_idToRegisters.size(), makeRegisterName(file->second, i)));
	}
}

}

}


