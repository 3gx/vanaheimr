/*! \file   Ray.h
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The header file for the Ray class 
*/

#include <baldr/include/XYZ.h>

namespace baldr
{
    class Ray
    {
        public:
            Ray(XYZ coord, XYZ S0) : m_equation(e), m_startingPoint(S0);
            XYZ getRayEquation() { return this->m_coordinates; };
            float getRayStart() { return this->m_startingPoint; };
        private:
            XYZ m_equation;
            XYZ m_startingPoint;
    }
}
