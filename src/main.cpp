#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include "controller.hpp"


using namespace std;
using namespace cv;

#define f 3740	//distância focal em pixels
#define b 160	//baseline em mm



Mat disp, disp2, Rview, Lview, normalized_disp, depth;

// mouseClick é responsável por identificar as coordenadas do pixel desejado. 
void mouseClick(int event, int x, int y, int flags, void* userdata){
	int i;
    Controller *controller = (Controller*)userdata;

    if  ( event == EVENT_LBUTTONDOWN ){
    	if(controller->clicks < 10){
    		Vec2f point = (x,y);
    		controller->points.push_back(point);
    		++controller->clicks;
    	}
    	cout << controller->windowsName << " " << controller->points.size() << endl;
    }

}

void requisito2(){
	Controller Lview, Rview;

	Lview.image = imread("../data/MorpheusL.jpg");
	Rview.image = imread("../data/MorpheusR.jpg");

	if(!Lview.image.data || !Rview.image.data){
        cout << endl << "Nao foi possivel abrir imagem ou ela não está nessa pasta!" << endl;
        while (cin.get() != '\n');
        return;
    }

    Lview.windowsName = "Lview";
    Rview.windowsName = "Rview";

    namedWindow(Lview.windowsName, WINDOW_NORMAL);
	imshow(Lview.windowsName, Lview.image);

	namedWindow(Rview.windowsName, WINDOW_NORMAL);
	imshow(Rview.windowsName, Rview.image);

	setMouseCallback(Lview.windowsName, mouseClick, &Lview);

	setMouseCallback(Rview.windowsName, mouseClick, &Rview);

    while(Lview.points.size() < 10 || Rview.points.size() < 10){
   		waitKey(100);
	}

	vector<cv::Vec3f> lines1, lines2;

	Mat F = findFundamentalMat(Rview.points, Lview.points, FM_RANSAC, 3, 0.99);

	Size size;

	size = F.size();

	cout << size;
	/*computeCorrespondEpilines(Rview.points, 1, F, lines1);
	computeCorrespondEpilines(Lview.points, 2, F, lines2);


	Scalar color(255,255,255);
 	for(int i=0; i<lines1.size(); i++){
	    line(Lview.image,Point(0,-lines1[i][2]/lines1[i][1]),Point(Lview.image.cols,-(lines1[i][2]+lines1[i][0]*Lview.image.cols)/lines1[i][1]),color);

	 	line(Rview.image,Point(0,-lines2[i][2]/lines2[i][1]),Point(Rview.image.cols,-(lines2[i][2]+lines2[i][0]*Rview.image.cols)/lines2[i][1]),color);

	}
*/
	namedWindow(Lview.windowsName, WINDOW_NORMAL);
	imshow(Lview.windowsName, Lview.image);

	namedWindow(Rview.windowsName, WINDOW_NORMAL);
	imshow(Rview.windowsName, Rview.image);

}


vector<float> getWorldCoordinates(float xl, float xr, float yl, float yr){
	float X,Y,Z;
	float max;
	vector<float> objectPoint;

	if( (xl-xr) == 0){
			X=-1;
			Y=-1;
			Z=-1;
	}
	else{
		X = ( b*(xl+xr) )/( 2*(xl-xr) );
		Y = ( b*(yl+yr) )/( 2*(xl-xr) );
		Z = ( b*f )/( 2*(xl-xr) );
	}

	objectPoint.push_back(X);
	objectPoint.push_back(Y);
	objectPoint.push_back(Z);
	return objectPoint;
}


void createDepth(){
	int min = 10000000, max = 0;

	depth = Mat(disp.rows, disp.cols, CV_32S);
	Mat normalized_depth;
	float xl, xr, yl, yr;
	int Z;

	for(int j = 0; j < disp.rows; ++j){
		for(int i = 0; i < disp.cols; ++i){

			xl = i;
			xr = i - (int)disp2.at<uchar>(j, i);
			yl = j;
			yr = yl;
			vector<float> objectPoint = getWorldCoordinates(xl, xr, yl, yr);
			depth.at<int>(j,i) = objectPoint[2];
			if(objectPoint[2] < min){
				min = objectPoint[2];
			}
			if(objectPoint[2] > max){
				max = objectPoint[2];
			}
		}
	}
	namedWindow("Profundidade2", WINDOW_NORMAL);
	imshow("Profundidade2", depth);
	namedWindow("disp", WINDOW_NORMAL);
	imshow("disp", disp2);
	cout << "min = "<< min << " max = " << max << endl;
	normalized_depth = Mat(disp.rows, disp.cols, CV_8U);

	for(int j = 0; j<disp.rows; ++j){
		for(int i = 0; i<disp.cols; ++i){
			normalized_depth.at<char>(j,i) = depth.at<int>(j,i)*255/(max-min) - 255*min/(max-min);
		}
	}

	//minMaxLoc(depth, &min, &max);
	cout << "min = "<< min << " max = " << max << endl;

	//depth.convertTo(normalized_depth, CV_16S, -255/(max-min), 255*min/(max-min));
	namedWindow("Profundidade", WINDOW_NORMAL);
	imshow("Profundidade", normalized_depth);

	;
	//minMaxLoc(normalized_depth, &min, &max);
	cout << "min = "<< min << " max = " << max << endl;

	waitKey(0);
}


void createDisparity(){
	int minDisparity = 10;
	int numDisparities = 128;
	int SADWindowSize = 5;
	int P1 = 600;
	int P2 = 2400;
	int disp12MaxDiff = 20;
	int preFilterCap = 16;
	int uniquenessRatio = 5;
	int speckleWindowSize = 100;
	int speckleRange = 20;
	double min, max;


	Rview = imread("../data/babyR.png");
	Lview = imread("../data/babyL.png");

	cvtColor(Lview, Lview,COLOR_RGB2GRAY, 0);
	cvtColor(Rview, Rview,COLOR_RGB2GRAY, 0);

	Ptr<StereoSGBM> stereo_sgbm = StereoSGBM::create(minDisparity, numDisparities, SADWindowSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, false);
	stereo_sgbm->compute(Lview,Rview,disp);

	// A saida do disparity map é uma matriz em que cada elemento possui 16bits sinalizados, sendo os 4 ultimos fracionais
	// Vamos ignorar os bits fracionais.
	disp2 = Mat(disp.rows, disp.cols, CV_8U);

	for(int j = 0; j < disp.rows ; j++){
		for(int i = 0; i < disp.cols ; i++){
			short int m; 
			m = disp.at<short int>(j, i);
			m = m/16; // divide por 16 para ignorar os últimos 4 bits
			disp2.at<char>(j,i) = m;
		}
	}

	//Normalizacao
	minMaxLoc(disp2, &min, &max);
	cout << "min = "<< min << " max = " << max << endl;
	disp2.convertTo(normalized_disp, CV_8U, 255/(max-min), -255*min/(max-min));

	imwrite("../data/aloe_disp.png", normalized_disp);
	namedWindow("Disparidade", WINDOW_NORMAL);
	imshow("Disparidade", normalized_disp);
	namedWindow("Real", WINDOW_NORMAL);
	imshow("Real", disp2);
	waitKey(0);

	destroyAllWindows();
}



Mat EncontraCorrelacao(Mat imgL, Mat disp){

	Mat corr;		//Matriz que possui a coordenada x da outra imagem correspondente a cada pixel da imagem atual

	for(int j = 0; j < imgL.rows ; j++){
		for(int i = 0; i < imgL.cols ; i++){
			corr.at<uchar>(j, i) = i + (int)disp.at<uchar>(j, i);
		}
	}

	return corr;

}


//encontra o maior valor de distância e transforma pra 255, e os outros valores ficam proporcionais a isso
void encontraMaioreTransforma(Mat* image){

	int i,j;
	int m = 0;

	for(j = 0; j < (*image).rows ; j++){
		for(i = 0; i < (*image).cols ; i++){
			if(m < (int)(*image).at<uchar>(j, i))
				m = (int)(*image).at<uchar>(j, i);
		}
	}
	cout << m << endl;

	for(j = 0; j < (*image).rows ; j++){
		for(i = 0; i < (*image).cols ; i++){
			(*image).at<uchar>(j, i) = (*image).at<uchar>(j, i)*((int) 255/m);
		}
	}			

}

void requisito1(){
	createDisparity();
	createDepth();
}


int main(){
	requisito2();
}

