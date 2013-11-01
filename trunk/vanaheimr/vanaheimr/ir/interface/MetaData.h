/*!	\file   MetaData.h
	\date   Wednesday October 30, 2013
	\author Gregory Diamos
	\brief  The header file for the MetaData class.
*/

#pragma once

namespace vanaheimr
{

namespace ir
{

/*! \brief A class for general purpose IR metadata */
class MetaData
{
public:
	virtual ~MetaData();

public:
	virtual MetaData* clone() const = 0;

};

}

}


