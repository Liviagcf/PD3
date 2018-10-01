/*  Aluna: Natalia Oliveira Borges
    Matricula: 16/0015863
    Projeto Demonstrativo 1 - Explorando OpenCV 
*/

#include "imageClass.hpp"

imageClass::imageClass(){
	isRGB = true;
	click = false; 
}

imageClass::~imageClass(){}

void imageClass::setImage(cv::Mat im){
	image = im;
}
void imageClass::setRequisito(int req){
	requisito = req;
}

void imageClass::setPixel(cv::Vec3b pix){
	pixel = pix;
}

cv::Mat imageClass::getImage(){
	return image;
}
int imageClass::getRequisito(){
	return requisito;
}
cv::Vec3b imageClass::getPixel(){
	return pixel;
}