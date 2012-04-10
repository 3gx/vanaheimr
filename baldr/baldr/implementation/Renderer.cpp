/*! \file   Renderer.cpp
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The implementation file for the Renderer class 
        
*/

// Standard Library Includes
#include <iostream>
//other includes
#include <baldr/include/Renderer.h> 
//Forward declarations

namespace baldr
{
    void Renderer::renderScene()
    {
        XYZ sampleOffsetX = (m_viewport.m_dy.getRayEquation()).scalarDivide(m_width);
        XYZ sampleOffsetY = (m_viewport.m_dy.getRayEquation()).scalarDivide(m_height);

        for (unsigned h = 0; h < m_height; ++h)
        {
            for (unsigned w = 0; w < m_width; ++w)
            {
                XYZ currentPixOffsetY = sampleOffsetY.scalarProduct(h);
                XYZ currentPixOffsetX = sampleOffsetX.scalarProduct(w);
                XYZ currentPos        = (currentPixOffsetX.add(currentPixOffsetY)).add(m_viewport.m_corner);
                Ray testRay(currentPos, m_viewport.m_corner);

                for (ObjectsInScene::iterator obj = m_objects.begin(); obj != m_objects.end(); ++obj)
                {
                    if (obj->doesIntersect(testRay))
                    {
                        std::cout << "#";
                    }
                    else
                    {
                        std::cout << " ";
                    }
                }

            }
            std::cout << "\n";
        }
    }
}
