#include <opencv2/opencv.hpp>

class Controller{

public:
   	Controller();
	~Controller();

 	std::string windowsName;
 	cv::Mat image;
 	std::vector<cv::Vec2f> points;
 	int clicks;
 	cv::Mat intrinsics, rotation, translation, extrinsics, projection;
 	std::vector<float> distcoef;

};