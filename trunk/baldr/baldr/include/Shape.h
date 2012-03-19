/*! \file   Shape.h
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The header file for the Shape class which is base of all 3D objects.
        
*/

// Standard Library Includes

// Forward Declarations

namespace baldr
{
namespace SceneObjects
{
    class Shape
    {
        bool doesIntersectionTest(Ray R) = 0;
    }//class Shape ends
}//SceneObjects ends
}
