/*!	\file Timer.h
*
*	\brief Header file for the Timer class
*
*	\author Gregory Diamos
*
*	\date : 9/27/2007
*
*/

#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED

#include <gpu-native/util/interface/LowLevelTimer.h>
#include <string>

/*!
	\brief a namespace for hydrazine classes and functions
*/
namespace gpunative
{

namespace util
{

	class Timer : public LowLevelTimer
	{
		public:	
			std::string toString() const;
	};

}

}

#endif

