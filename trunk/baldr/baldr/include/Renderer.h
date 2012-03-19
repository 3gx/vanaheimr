/*! \file   Renderer.h
        \date   Sunday March 18, 2012
        \author Sudnya Diamos <mailsudnya@gmail.com>
        \brief  The header file for the Renderer class 
        
*/

// Standard Library Includes

//other includes
//Forward declarations

namespace baldr
{
    class Viewport
    {
        public:
            Viewport(XYZ dx, XYZ dy, XYZ corner) : m_dx(dx, corner), m_dy(dy, corner), m_corner(corner);
        private:
            Ray m_dx, m_dy;
            XYZ m_corner;
    }

    class Renderer
    {
        public:
            typedef std::vector<Shape> ObjectsInScene;

            Renderer(XYZ camera, Viewport viewport, unsigned width, unsigned height) :
                m_camera(camera), m_viewport(viewport), m_width(width), m_height(height);
            void AddObjectToScene(Shape s) { m_objects.push_back(s); };

            void renderScene();

        private:
            ObjectsInScene m_objects;
            XYZ m_camera;
            Viewport m_viewport;
            unsigned m_width, m_height;
    }
}
