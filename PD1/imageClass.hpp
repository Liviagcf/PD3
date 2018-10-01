/*  Aluna: Natalia Oliveira Borges
    Matricula: 16/0015863
    Projeto Demonstrativo 1 - Explorando OpenCV 
*/

#include <opencv2/opencv.hpp>

class imageClass{

public:
    imageClass();
	~imageClass();

	void setImage(cv::Mat im);
	void setRequisito(int req);
	void setPixel(cv::Vec3b pix);
	cv::Mat getImage();
	int getRequisito();
	cv::Vec3b getPixel();
	bool isRGB;
	bool click;


private:
    cv::Mat image;
    int requisito; 
    cv::Vec3b pixel; 
};