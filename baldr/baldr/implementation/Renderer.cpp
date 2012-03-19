/*! \file   Renderer.cpp
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The implementation file for the Renderer class 
        
*/

// Standard Library Includes

//other includes
//Forward declarations

namespace baldr
{
    void renderScene()
    {
        XYZ sampleOffsetX = (m_dx.getRayEquation()).scalarDivide(m_width);
        XYZ sampleOffsetY = (m_dy.getRayEquation()).scalarDivide(m_height);

        for (unsigned h = 0; h < m_height; ++h)
        {
            for (unsigned w = 0; w < m_width; ++w)
            {
                XYZ currentPixOffsetY = sampleOffsetY.scalarProduct(h);
                XYZ currentPixOffsetX = sampleOffsetX.scalarProduct(w);
                XYZ currentPos        = (currentPixOffsetX.add(currentPixOffsetY)).add(m_corner);
                Ray testRay(currentPos, m_corner);

                for (ObjectsInScene::iterator obj = m_objects.begin(); obj != m_objects.end(); ++obj)
                {
                    if (doesIntersect(testRay))
                    {
                        std::cout << "#";
                    }
                }

            }
            std::cout << "\n";
        }
    }
}
