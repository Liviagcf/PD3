#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <fstream>
#include <math.h>


using namespace std;
using namespace cv;

#define f 3740	//distância focal em pixels
#define b 160	//baseline em mm


void calculaCoordenadas(int xl, int xr, int yl, int yr){
	double X,Y,Z;

	X = ( b*(xl+xr) )/( 2*(xl-xr) );
	Y = ( b*(yl+yr) )/( 2*(xl-xr) );
	Z = ( b*f )/( 2*(xl-xr) );

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


void createDisparity(){
	int minDisparity = 0;
	int numDisparities = 128;
	int SADWindowSize = 5;
	int P1 = 600;
	int P2 = 2400;
	int disp12MaxDiff = 20;
	int preFilterCap = 16;
	int uniquenessRatio = 1;
	int speckleWindowSize = 100;
	int speckleRange = 20;
	double min, max;

	Mat aloeR, aloeL, babyR, babyL, disp, normalized_disp;


	aloeR = imread("../data/aloeR.png");
	aloeL = imread("../data/aloeL.png");
	babyR = imread("../data/babyR.png");
	babyL = imread("../data/babyL.png");

	cvtColor(aloeL, aloeL,COLOR_RGB2GRAY, 0);
	cvtColor(aloeR, aloeR,COLOR_RGB2GRAY, 0);

	Ptr<StereoSGBM> stereo_sgbm = StereoSGBM::create(minDisparity, numDisparities, SADWindowSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, false);
	stereo_sgbm->compute(aloeL,aloeR,disp);

	//Normalizacao
	minMaxLoc(disp, &min, &max);
	disp.convertTo(normalized_disp, CV_8U, 255/(max-min), -255*min/(max-min));

	imwrite("../data/aloe_disp.png", normalized_disp);

	namedWindow("Disparidade", WINDOW_NORMAL);
	imshow("Disparidade", normalized_disp);
	namedWindow("Real", WINDOW_NORMAL);
	imshow("Real", disp);


	waitKey(0);


}

void requisito1(){
	/*
	Mat aloeR, aloeL, babyR, babyL, right_disp, left_disp, ldisp, rdisp;
	int N_DISP = 64, W = 5;
	Ptr<StereoBM> stereo;

	aloeR = imread("../data/aloeR.png");
	aloeL = imread("../data/aloeL.png");
	babyR = imread("../data/babyR.png");
	babyL = imread("../data/babyL.png");

	/*cout << "Digite o tamanho da janela: ";s
	cin >> W;
	if(W < 5)
		W = 5;
	else if(W%2 == 0)
		W++;


	//Parece que calcula a distância dos pixels entre as duas imagens
	cvtColor(aloeL, aloeL,COLOR_RGB2GRAY, 0);
	cvtColor(aloeR, aloeR,COLOR_RGB2GRAY, 0);

	stereo = StereoBM::create(N_DISP, W);  //não sei muito bem o que o sengundo parâmetro faz, mas muda algumas coisas na matriz disp
	stereo->compute(aloeL,aloeR,left_disp);
	//stereo->compute(aloeR, aloeL, right_disp);

	normalize(left_disp, ldisp, 0, 256, cv::NORM_MINMAX, CV_8U);	
	normalize(right_disp, rdisp, 0, 256, cv::NORM_MINMAX, CV_8U);

	imwrite("../data/aloe_disp.png", ldisp);

	namedWindow("Disparidade", WINDOW_NORMAL);
	namedWindow("Disparidade2", WINDOW_NORMAL);
	imshow("Disparidade", ldisp);
	//imshow("Disparidade2", rdisp);

	waitKey(0);*/

	createDisparity();

}

void requisito2(){
	/*cv::Mat R1, R2, P1, P2, Q;
	stereoRectify(K1, D1, K2, D2, img1.size(), R, T, R1, R2, P1, P2, Q);
	https://sourishghosh.com/2016/stereo-calibration-cpp-opencv/ */
}




int main(){
	requisito1();
}

