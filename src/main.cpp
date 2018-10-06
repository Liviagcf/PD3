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
    		Point2f point = Point2f(x,y);
    		controller->points.push_back(point);
    		++controller->clicks;
    	}
    	cout << controller->windowsName << " " << controller->points.size() << endl;
    }

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


	//Rview = imread("Limage.jpg");
	//Lview = imread("Rimage.jpg");

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

void requisito2(){
	Controller Lcamera, Rcamera;

	//Inicializando os dados
	Lcamera.image = imread("../data/MorpheusL.jpg");
	Rcamera.image = imread("../data/MorpheusR.jpg");

	if(!Lcamera.image.data || !Rcamera.image.data){
        cout << endl << "Nao foi possivel abrir imagem ou ela não está nessa pasta!" << endl;
        while (cin.get() != '\n');
        return;
    }
    Lcamera.windowsName = "Lcamera";
    Rcamera.windowsName = "Rcamera";

    double data1[9] = {6704.926882, 0.000103, 738.251932, 0, 6705.241311, 457.560286, 0, 0, 1};
    double data2[9] = {6682.125964, 0.000101, 875.207200, 0, 6681.475962, 357.700292, 0, 0, 1};
    double data3[9] = {0.70717199,  0.70613396, -0.03581348, 0.28815232, -0.33409066, -0.89741388 ,-0.64565936,  0.62430623, -0.43973369};
    double data4[9] = {0.48946344,  0.87099159, -0.04241701 ,0.33782142, -0.23423702, -0.91159734 ,-0.80392924,  0.43186419, -0.40889007};
    double data5[3] = {-532.285900 , 207.183600 , 2977.408000};
    double data6[3] = {-614.549000 , 193.240700 , 3242.754000}; 


    Lcamera.intrinsics = Mat(3,3,CV_64FC1, data1);
   	Rcamera.intrinsics = Mat(3,3,CV_64FC1, data2);

    Lcamera.rotation  = Mat(3,3,CV_64FC1, data3);
    Rcamera.rotation  = Mat(3,3,CV_64FC1, data4);

    Lcamera.translation = Mat(3,1,CV_64FC1, data5);
    Rcamera.translation =Mat(3,1,CV_64FC1, data6);

    Mat Linverse;
	invert(Lcamera.rotation, Linverse, DECOMP_LU);

    //Rotacao da camera da direita em relacao a camera da esquerda  R = Rl.inv * Rr,
 	//Translacao da camera da direita em relacao a da esquerda T = Rl.inv * (tr - tl).
    Mat R = Linverse*Rcamera.rotation;
    Mat T = Linverse*(Rcamera.translation - Lcamera.translation);
    
    namedWindow(Lcamera.windowsName, WINDOW_NORMAL);
	imshow(Lcamera.windowsName, Lcamera.image);

	namedWindow(Rcamera.windowsName, WINDOW_NORMAL);
	imshow(Rcamera.windowsName, Rcamera.image);

	//setMouseCallback(Lcamera.windowsName, mouseClick, &Lcamera);

	//setMouseCallback(Rcamera.windowsName, mouseClick, &Rcamera);

	waitKey(0);

   /* while(Lcamera.points.size() < 1 || Rcamera.points.size() < 1){
   		waitKey(100);
	}*/

	cout<<"saiu"<<endl;

	//Imagens precisam ter o mesmo tamanho
	resize(Rcamera.image, Rcamera.image, Lcamera.image.size(),6682.125964*0.000101 ,6681.475962*0.000101, INTER_LINEAR); 
	imshow(Rcamera.windowsName, Rcamera.image);
	waitKey(0);

	Mat R1, R2, P1, P2, Q, LMap1, LMap2, RMap1, RMap2, Lout, Rout;

	stereoRectify(Lcamera.intrinsics, Lcamera.distcoef, Rcamera.intrinsics, Rcamera.distcoef, Lcamera.image.size(), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, Lcamera.image.size(), 0, 0 );

	initUndistortRectifyMap(Lcamera.intrinsics, Lcamera.distcoef, R1, P1, Lcamera.image.size(),CV_32FC1, LMap1, LMap2);
	initUndistortRectifyMap(Rcamera.intrinsics, Rcamera.distcoef, R2, P2, Lcamera.image.size(),CV_32FC1, RMap1, RMap2);

	remap(Lcamera.image, Lout, LMap1, LMap2, INTER_LINEAR, BORDER_CONSTANT, 0);
	remap(Rcamera.image, Rout, RMap1, RMap2, INTER_LINEAR, BORDER_CONSTANT, 0);

	imshow(Lcamera.windowsName, Lout);
	imshow(Rcamera.windowsName, Rout);

	waitKey(0);




/*
	//Calcula epilines (funciona parcialmente)
	vector<cv::Vec3f> lines1, lines2;

	Mat F = findFundamentalMat(Lcamera.points, Rcamera.points, FM_RANSAC, 3, 0.99);

	Mat H1, H2;

	//H1 = findHomography(Lcamera.points, Rcamera.points, CV_RANSAC, 3);
	stereoRectifyUncalibrated(Lcamera.points, Rcamera.points, F, Lcamera.image.size(), H1, H2, 5);

	Mat Ldst, Rdst;

	warpPerspective(Rcamera.image, Rdst, H1, Rcamera.image.size(), INTER_LINEAR , BORDER_TRANSPARENT);
	//warpPerspective(Lcamera.image, Ldst, H2, Rcamera.image.size());

	computeCorrespondEpilines(Rcamera.points, 1, F, lines1);
	computeCorrespondEpilines(Lcamera.points, 2, F, lines2);


	Scalar color(0,0,255);
 	for(int i=0; i<lines1.size(); i++){
	    line(Lcamera.image,Point(0,-lines1[i][2]/lines1[i][1]),Point(Lcamera.image.cols,-(lines1[i][2]+lines1[i][0]*Lcamera.image.cols)/lines1[i][1]),color);

	 	line(Rcamera.image,Point(0,-lines2[i][2]/lines2[i][1]),Point(Rcamera.image.cols,-(lines2[i][2]+lines2[i][0]*Rcamera.image.cols)/lines2[i][1]),color);
	}


	namedWindow(Lcamera.windowsName, WINDOW_NORMAL);
	imshow(Lcamera.windowsName, Lcamera.image);

	namedWindow(Rcamera.windowsName, WINDOW_NORMAL);
	imshow(Rcamera.windowsName, Rdst);

	imwrite("Limage.jpg", Lcamera.image);
	imwrite("Rimage.jpg", Rdst);

	cout << Lcamera.image.size() << " " << Rdst.size() << " " << Rcamera.image.size() <<endl;

	Lview=Lcamera.image;
	Rview=Rdst;


	waitKey(0);
	//createDisparity();*/
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

