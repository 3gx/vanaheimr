/*!	\file   ImageFile.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday, April 8, 2012
	\brief  The source file for the ImageFile class.
*/


// Baldr Includes
#include <baldr/include/ImageFile.h>


namespace baldr
{

ImageFile::ImageFile(unsigned int width, unsigned int height);

void ImageFile::clear();
void ImageFile::resize(unsigned int width, unsigned int height);
void ImageFile::setPixel(unsigned int x, unsigned int y,
	unsigned int red, unsigned int green,
	unsigned int blue, unsigned int alpha);

void ImageFile::write(const std::string& filename);
unsigned int ImageFile::_getIndex(unsigned int x, unsigned int y);


}


