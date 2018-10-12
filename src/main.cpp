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
    	if(controller->clicks < 30){
    		Point2f point = Point2f(x,y);
    		controller->points.push_back(point);
    		++controller->clicks;
    	}
    	//cout << x << ' ' << y << endl;
    	cout << controller->windowsName << " " << controller->points.size() << endl;
    }

}


vector<float> getWorldCoordinates(float xl, float xr, float yl, float yr){
	float X,Y,Z;
	float max;
	vector<float> objectPoint;

	X = ( b*(xl+xr) )/( 2*(xl-xr) );
	Y = ( b*(yl+yr) )/( 2*(xl-xr) );
	Z = ( b*f )/( 2*(xl-xr) );
	
	objectPoint.push_back(X);
	objectPoint.push_back(Y);
	objectPoint.push_back(Z);
	return objectPoint;
}


void createDepth(){
	int min = 10000000, max = 0;

	cout << "Calculando profundidade..." << endl;

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

	normalized_depth = Mat(disp.rows, disp.cols, CV_8U);

	for(int j = 0; j<disp.rows; ++j){
		for(int i = 0; i<disp.cols; ++i){
			normalized_depth.at<char>(j,i) = depth.at<int>(j,i)*255/(max-min) - 255*min/(max-min);
		}
	}

	namedWindow("Profundidade", WINDOW_NORMAL);
	imshow("Profundidade", normalized_depth);
	imwrite("profundidade.png", normalized_depth);
	cout << "Aperte uma tecla para continuar! " << endl;
	waitKey(0);
}


void createDisparity(){
	cout << "Calculando disparidade ..." << endl;

	int minDisparity = 1;
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
	disp2.convertTo(normalized_disp, CV_8U, 255/(max-min), -255*min/(max-min));

	imwrite("../data/disparidade.png", normalized_disp);
	namedWindow("Disparidade", WINDOW_NORMAL);
	imshow("Disparidade", normalized_disp);
	cout << "Aperte uma tecla para continuar! " << endl;
	waitKey(0);

	destroyAllWindows();
}

void stereoRetification(){
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
    namedWindow(Lcamera.windowsName, WINDOW_NORMAL);
	namedWindow(Rcamera.windowsName, WINDOW_NORMAL);

    double data1[9] = {6704.926882, 0.000103, 738.251932, 0, 6705.241311, 457.560286, 0, 0, 1};
    double data2[9] = {6682.125964, 0.000101, 875.207200, 0, 6681.475962, 357.700292, 0, 0, 1};
    double data3[9] = {0.70717199,  0.70613396, -0.03581348, 0.28815232, -0.33409066, -0.89741388 ,-0.64565936,  0.62430623, -0.43973369};
    double data4[9] = {0.48946344,  0.87099159, -0.04241701, 0.33782142, -0.23423702, -0.91159734 ,-0.80392924,  0.43186419, -0.40889007};
    double data5[3] = {-532.285900 , 207.183600 , 2977.408000};
    double data6[3] = {-614.549000 , 193.240700 , 3242.754000}; 


    Lcamera.intrinsics = Mat(3,3,CV_64FC1, data1);
   	Rcamera.intrinsics = Mat(3,3,CV_64FC1, data2);

    Lcamera.rotation  = Mat(3,3,CV_64FC1, data3);
    Rcamera.rotation  = Mat(3,3,CV_64FC1, data4);

    Lcamera.translation = Mat(3,1,CV_64FC1, data5);
    Rcamera.translation = Mat(3,1,CV_64FC1, data6);

    Mat Linverse;
	invert(Lcamera.rotation, Linverse, DECOMP_LU);

    //Rotacao da camera da direita em relacao a camera da esquerda  R = Rl.inv * Rr,
 	//Translacao da camera da direita em relacao a da esquerda T = Rl.inv * (tr - tl).
    Mat R = Linverse*Rcamera.rotation;
    Mat T = Linverse*(Rcamera.translation-Lcamera.translation);
    
	resize(Rcamera.image, Rcamera.image, Lcamera.image.size(),6682.125964*0.000101 ,6681.475962*0.000101, INTER_LINEAR); 
	imshow(Rcamera.windowsName, Rcamera.image);
	imshow(Lcamera.windowsName, Lcamera.image);
	cout<<"Pressione uma tecla " << endl;
	waitKey(0);

	Mat R1, R2, P1, P2, Q, LMap1, LMap2, RMap1, RMap2, Lout, Rout;

	stereoRectify(Lcamera.intrinsics, Lcamera.distcoef, Rcamera.intrinsics, Rcamera.distcoef, Lcamera.image.size(), R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, Lcamera.image.size(), 0,0 );

	initUndistortRectifyMap(Lcamera.intrinsics, Lcamera.distcoef, R1, P1, Lcamera.image.size(),CV_32FC1, LMap1, LMap2);
	initUndistortRectifyMap(Rcamera.intrinsics, Rcamera.distcoef, R2, P2, Lcamera.image.size(),CV_32FC1, RMap1, RMap2);

	remap(Lcamera.image, Lout, LMap1, LMap2, INTER_LINEAR, BORDER_CONSTANT, 0);
	remap(Rcamera.image, Rout, RMap1, RMap2, INTER_LINEAR, BORDER_CONSTANT, 0);

	imshow(Lcamera.windowsName, Lout);
	imshow(Rcamera.windowsName, Rout);

	waitKey(0);
	destroyAllWindows();
}


void uncalibratedRetification(){
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

    resize(Rcamera.image, Rcamera.image, Lcamera.image.size(),6682.125964*0.000101 ,6681.475962*0.000101, INTER_LINEAR);

	namedWindow(Lcamera.windowsName, WINDOW_NORMAL);
	imshow(Lcamera.windowsName, Lcamera.image);
	namedWindow(Rcamera.windowsName,WINDOW_NORMAL);
	imshow(Rcamera.windowsName, Rcamera.image);
	setMouseCallback(Lcamera.windowsName, mouseClick, &Lcamera);
	setMouseCallback(Rcamera.windowsName, mouseClick, &Rcamera);

	cout << "Clique em 20 pontos correspondentes na imagem." << endl;

    while(Lcamera.points.size() < 20 || Rcamera.points.size() < 20){
   		waitKey(100);
	}

	//Calcula epilines (funciona parcialmente) 
	vector<cv::Vec3f> lines1, lines2;
	Mat H1, H2;
	Mat F = findFundamentalMat(Rcamera.points, Lcamera.points, FM_RANSAC, 3, 0.99);
	stereoRectifyUncalibrated(Rcamera.points, Lcamera.points, F, Lcamera.image.size(), H1, H2, 5);

	Mat Ldst, Rdst;

	computeCorrespondEpilines(Rcamera.points, 1, F, lines1);
	computeCorrespondEpilines(Lcamera.points, 2, F, lines2);
	Rdst = Rcamera.image.clone();
	Ldst = Lcamera.image.clone();

 	for(int i=0; i<lines1.size(); i++){
 		Scalar color(rand() % 255,rand() % 255,rand() % 255);  
	    line(Ldst, Point(0,-lines1[i][2]/lines1[i][1]), Point(Lcamera.image.cols,-(lines1[i][2]+lines1[i][0]*Lcamera.image.cols)/lines1[i][1]),color, 4,8,0);
	 	line(Rdst,Point(0, -lines2[i][2]/lines2[i][1]), Point(Rcamera.image.cols,-(lines2[i][2]+lines2[i][0]*Rcamera.image.cols)/lines2[i][1]),color,4,8,0);	
		circle(Ldst, Point(Lcamera.points[i][0],Lcamera.points[i][1]), 15, color, 10, 8, 0);
		circle(Rdst, Point(Rcamera.points[i][0],Rcamera.points[i][1]), 15, color, 10, 8, 0);
	}

	imwrite("epipolarL.png", Ldst);
	imwrite("epipolarR.png", Rdst);

	namedWindow(Rcamera.windowsName, WINDOW_NORMAL);
	imshow(Rcamera.windowsName, Rdst);
	namedWindow(Lcamera.windowsName, WINDOW_NORMAL);
	imshow(Lcamera.windowsName, Ldst);

	waitKey(0);

	warpPerspective(Rcamera.image, Rcamera.image, H1, Rcamera.image.size(), INTER_LINEAR , BORDER_CONSTANT);
	warpPerspective(Lcamera.image, Lcamera.image, H2, Rcamera.image.size(),INTER_LINEAR , BORDER_CONSTANT);
	warpPerspective(Rdst, Rdst, H1, Rcamera.image.size(), INTER_LINEAR , BORDER_CONSTANT);
	warpPerspective(Ldst, Ldst, H2, Rcamera.image.size(),INTER_LINEAR , BORDER_CONSTANT);

	imwrite("rectifiedepipolarL.png", Ldst);
	imwrite("rectifiedepipolarR.png", Rdst);

	namedWindow(Lcamera.windowsName, WINDOW_NORMAL);
	imshow(Lcamera.windowsName, Lcamera.image);
	namedWindow(Rcamera.windowsName, WINDOW_NORMAL);
	imshow(Rcamera.windowsName, Rcamera.image);

	waitKey(0);

	imwrite("rectifiedL.png", Lcamera.image);
	imwrite("rectifiedR.png", Rcamera.image);

	destroyAllWindows();

	Lview = Lcamera.image.clone();
	Rview = Rcamera.image.clone();

	createDisparity();
	createDepth();
}

void homography(){
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
    namedWindow(Lcamera.windowsName, WINDOW_NORMAL);
    namedWindow(Rcamera.windowsName, WINDOW_NORMAL);
    imshow(Lcamera.windowsName, Lcamera.image);
    imshow(Rcamera.windowsName, Rcamera.image);
    cout<<"Pressione uma tecla " << endl;
    waitKey();

	double data1[9] = {6704.926882, 0.000103, 738.251932, 0, 6705.241311, 457.560286, 0, 0, 1};
    double data2[9] = {6682.125964, 0.000101, 875.207200, 0, 6681.475962, 357.700292, 0, 0, 1};
    double data3[12] = {0.70717199,  0.70613396, -0.03581348,-532.285900, 0.28815232, -0.33409066, -0.89741388 ,207.183600,-0.64565936,  0.62430623, -0.43973369,2977.408000};
    double data4[12] = {0.48946344,  0.87099159, -0.04241701,-614.549000, 0.33782142, -0.23423702, -0.91159734 ,193.240700,-0.80392924,  0.43186419, -0.40889007,3242.754000};


    Lcamera.intrinsics = Mat(3,3,CV_64FC1, data1);
   	Rcamera.intrinsics = Mat(3,3,CV_64FC1, data2);

   	Lcamera.extrinsics = Mat(3,4,CV_64FC1, data3);
   	Rcamera.extrinsics = Mat(3,4,CV_64FC1, data4);

   	Mat P1 = Lcamera.intrinsics*Lcamera.extrinsics;
   	Mat P2 = Rcamera.intrinsics*Rcamera.extrinsics;

   	Mat P1transpost, P2transpost;
   	Mat P1seudoInverse, P2seudoInverse;
   	Mat H1, H2;
   	Mat dst1, dst2;


   	transpose(P1, P1transpost);
   	P1seudoInverse = P1transpost*P1;
   	invert( P1seudoInverse, P1seudoInverse, DECOMP_LU);

   	H1=P2*P1seudoInverse*P1transpost;


    warpPerspective(Lcamera.image, dst1, H1, Lcamera.image.size(),INTER_LINEAR , BORDER_TRANSPARENT);

   	transpose(P2, P2transpost);
   	P2seudoInverse = P2transpost*P2;
   	invert( P2seudoInverse, P2seudoInverse, DECOMP_LU);

   	H2=P1*P2seudoInverse*P2transpost;

    warpPerspective(Rcamera.image, dst2, H2, Lcamera.image.size(),INTER_LINEAR , BORDER_TRANSPARENT);

	imshow(Lcamera.windowsName, dst1);
	imshow(Rcamera.windowsName, dst2);
   	
   	waitKey();
   	destroyAllWindows();

}



float findVolume(vector<vector<float> > real){
	float volume, A,B,C;		// Arestas

	A = sqrt( (real[0][0]-real[1][0])*(real[0][0]-real[1][0]) + (real[0][1]-real[1][1])*(real[0][1]-real[1][1]) + (real[0][2]-real[1][2])*(real[0][2]-real[1][2]) );
	B = sqrt( (real[2][0]-real[1][0])*(real[2][0]-real[1][0]) + (real[2][1]-real[1][1])*(real[2][1]-real[1][1]) + (real[2][2]-real[1][2])*(real[2][2]-real[1][2]) );
	B = sqrt( (real[2][0]-real[3][0])*(real[2][0]-real[3][0]) + (real[2][1]-real[3][1])*(real[2][1]-real[3][1]) + (real[2][2]-real[3][2])*(real[2][2]-real[3][2]) );

	volume = A*B*C;

	cout << "A: " << real[0][0] << endl;
	cout << "B: " << real[0][1] << endl;
	cout << "C: " << real[0][2] << endl;

	return volume;
}


void requisito3(){


	Controller Rcamera, Lcamera;
	Lcamera.image = imread("../data/MorpheusL.jpg");
	Rcamera.image = imread("../data/MorpheusR.jpg");

	Lcamera.windowsName = "Lcamera";
    Rcamera.windowsName = "Rcamera";

	vector<vector<float> > real;
	Point point(0,0);
	float xl,yl,xr,yr, zl;
	int i;

	namedWindow(Lcamera.windowsName, WINDOW_NORMAL);
	imshow(Lcamera.windowsName, Lcamera.image);

	namedWindow(Rcamera.windowsName, WINDOW_NORMAL);
	imshow(Rcamera.windowsName, Rcamera.image);

	setMouseCallback(Rcamera.windowsName, mouseClick, &Rcamera);
	setMouseCallback(Lcamera.windowsName, mouseClick, &Lcamera);

	float data1[12] = {6704.926882, 0.000103, 738.251932, 0, 0, 6705.241311, 457.560286, 0, 0, 0, 1, 0};
    float data2[12] = {6682.125964, 0.000101, 875.207200, 0, 0, 6681.475962, 357.700292, 0, 0, 0, 1, 0};

    Lcamera.intrinsics = Mat(3,4,CV_32FC1, data1);
   	Rcamera.intrinsics = Mat(3,4,CV_32FC1, data2);

   	Mat Pr_inv  = Rcamera.intrinsics.inv(DECOMP_SVD);

   	float baseline = sqrt( (738.251932 - 875.207200)*(738.251932 - 875.207200) + (457.560286 - 357.700292)*(457.560286 - 357.700292) );

    while(Rcamera.points.size() < 2 || Lcamera.points.size() < 2){
   		waitKey(100);
	}

	for(i=0; i<2; i++){
		xr = Rcamera.points[i][0];
		yr = Rcamera.points[i][1];
		xl = Lcamera.points[i][0];
		yl = Lcamera.points[i][1];

		/*float datar[3] = {xr, yr, 1};
		Mat R = Mat(3,1,CV_32FC1, datar);

		Mat L = (Lcamera.intrinsics*Pr_inv)*R;
		xl = L.at<float>(0,0);
		yl = L.at<float>(1,0);*/
		
		//Coloquei mais dois parâmetros pq a baseline e a distância focal são diferentes das outras imagens
		//real.push_back(getWorldCoordinates(xl, xr, yl, yr, baseline, 6705));
		cout << xl << endl;
		cout << xr << endl;
	}

	//cout << "Volume: " << findVolume(real) << endl;
}


void profundidade(){

	Mat disp = imread("../data/aloe_disp.png");
	Mat prof = disp.clone();
	int i,j;
	float min = 255, max = 0;
	Mat profi = Mat(disp.rows, disp.cols, CV_8U);


	for(i=0; i<disp.rows; i++){
		for(j=0; j<disp.cols; j++){
			prof.at<float>(i,j) = (b*f)/(2*(disp.at<float>(i,j)));
		}
	}

	for(int j = 0; j < prof.rows; ++j){
		for(int i = 0; i < prof.cols; ++i){

			if(prof.at<float>(i,j) < min){
				min = prof.at<float>(i,j);
			}
			if(prof.at<float>(i,j) > max){
				max = prof.at<float>(i,j);
			}
		}
	}

	for(int j = 0; j<disp.rows; ++j){
		for(int i = 0; i<disp.cols; ++i){
			int a;
			a = prof.at<float>(j,i);//*255/(max-min) - 255*min/(max-min);
			profi.at<uchar>(j,i) = a;
		}
	}


	namedWindow("Profundidade", WINDOW_NORMAL);
	imshow("Profundidade", prof);
	waitKey(0);

	destroyAllWindows();
}

void requisito1(){
	int escolha;
	system("clear");
	cout << "DISPARIDADE E PROFUNDIDADE DE IMAGENS RETIFICADAS" << endl << endl;
	cout << "Que imagens deseja abrir (1-aloe 2-baby):";
	cin >> escolha;

	if(escolha == 1){
		Lview = imread("../data/aloeL.png");
		Rview = imread("../data/aloeR.png");
	}

	if(escolha == 2){
		Lview = imread("../data/babyL.png");
		Rview = imread("../data/babyR.png");
	}

	if(!Lview.data || !Rview.data){
		cout << endl << "Não foi possível abrir o arquivo." << endl;
	}

	createDisparity();
	createDepth();
}

void requisito2(){
	system("clear");
	cout << "DISPARIDADE E PROFUNDIDADE DE IMAGENS NÃO RETIFICADAS" << endl << endl;
	uncalibratedRetification();

}

void menu(){
	int escolha = 0;
	while(escolha != 6){
		system("clear");
		cout << "IMAGENS STEREO" << endl << endl;
		cout << "1- Mapa de disparidade e profundidade de imagens retificadas" << endl;
		cout << "2- Mapa de disparidade e profundidade de imagens não retificadas" << endl;
		cout << "3- Cálculo de volume" << endl;
		cout << "4- Tentativa de retificação usando a função stereoRectify" << endl;
		cout << "5- Tentativa de retificação usando homografia" << endl;
		cout << "6- Sair" << endl;
		cout << "Sua escolha:";

		cin >> escolha;

		if(escolha == 1){
			requisito1();
		}

		if(escolha == 2){
			requisito2();
		}
		if(escolha == 3){
			
		}
		if(escolha == 4){
			stereoRetification();
		}
		if(escolha == 5){
			homography();
		}
		if(escolha == 6){
			break;
		}
	}

}


int main(){
	menu();
	
}

