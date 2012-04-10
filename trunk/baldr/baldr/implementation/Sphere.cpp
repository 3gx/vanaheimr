/*! \file   Sphere.cpp
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The implementation file for the Sphere class 
        
*/

// Standard Library Includes
#include <cmath>
//other includes
#include <baldr/include/Sphere.h>
// Forward Declarations

namespace baldr
{
namespace SceneObjects
{
    bool Sphere::doesIntersect(const Ray& R)
    {
        //start pt of ray is same as camera?
        float cameraToCentre = m_centre.distance(R.getRayStart());
        float distanceVector = std::sqrt((cameraToCentre*cameraToCentre) - (m_radius*m_radius));

        XYZ segmentAlongRay = (R.getRayEquation()).scalarProduct(distanceVector);
        XYZ pointInQuestion = R.getRayStart().add(segmentAlongRay);

        float distanceToPointInQuestion = pointInQuestion.distance(m_centre);

        return distanceToPointInQuestion > m_radius ? 0 : 1;
    }
}
}
