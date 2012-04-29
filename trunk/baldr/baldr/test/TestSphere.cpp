#include <baldr/include/Renderer.h>

int main()
{
    baldr::XYZ cameraPos(0, 0, 0);
    baldr::Viewport viewport(1, 1, 1);
    unsigned width = 2;
    unsigned height = 3;
    baldr::Renderer renderer(cameraPos, viewport, width, height);

    renderer.renderScene();
}
