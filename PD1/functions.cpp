/*  Aluna: Natalia Oliveira Borges
    Matricula: 16/0015863
    Projeto Demonstrativo 1 - Explorando OpenCV 
*/

#include "functions.hpp"
#include <math.h>

// getPixelValues é responsável por imprimir na tela os valores RGB ou grayscale do pixel clicado.
void getPixelValues(int row, int col, void* userdata){
    imageClass *imageData = (imageClass*)userdata;
    Vec3b pixel = imageData->getImage().at<Vec3b>(row, col);
    imageData->setPixel(pixel);

    if(imageData->isRGB){
        cout << "Blue = " << (int)pixel.val[0] << "  Green = " << (int)pixel.val[1] << "  Red = " << (int)pixel.val[2]  << endl << endl;
    }
    else{
        cout << "Gray = " << (int)pixel.val[0] << endl << endl;
    }    
}

// drawRedPixels é responsável por pintar os pixels 13 tons próximos do pixel clicado de vermelho.
void drawRedPixels(void* userdata, string windowName){
    imageClass *imageData = (imageClass*)userdata;
    int tom;
    Mat clone = imageData->getImage().clone();
    Vec3b originalPixel = imageData->getPixel();

    for(int i = 0; i < imageData->getImage().rows; ++i){
        for(int j = 0; j < imageData->getImage().cols; ++j){
            Vec3b pixel = imageData->getImage().at<Vec3b>(i, j);
            tom = pow((originalPixel.val[0]-pixel.val[0]),2) + pow((originalPixel.val[1]-pixel.val[1]),2) + pow((originalPixel.val[2]-pixel.val[2]),2);
            if(tom < 169){
                clone.at<Vec3b>(i, j)[0] = 0;
                clone.at<Vec3b>(i, j)[1] = 0;
                clone.at<Vec3b>(i, j)[2] = 255;              
            }
        }
    }

    imshow(windowName, clone);
}

// mouseClick é responsável por identificar as coordenadas do pixel desejado. 
void mouseClick(int event, int x, int y, int flags, void* userdata){
    imageClass *imageData = (imageClass*)userdata;

    if  ( event == EVENT_LBUTTONDOWN ){

        cout << "Row = " << y << "  Col = " << x << endl;

        getPixelValues(y, x, imageData);

        if(imageData->getRequisito() == REQUISITO2){
            drawRedPixels(imageData,"Image");
        }
        if(imageData->getRequisito() == REQUISITO3 || imageData->getRequisito() == REQUISITO4){
            imageData->click = true;
        }
    }
}

//Abre imagem para realizar os requisitos 1 e 2.
void image(int requisito){
    system("clear");
    Mat image;
    string image_name;

    cout << "Escreva o nome da imagem que quer usar (incluindo a extenção):";
    cin >> image_name;
    while (cin.get() != '\n');

    image = imread(image_name);

    if(!image.data){
        cout << endl << "Nao foi possivel abrir imagem ou ela não está nessa pasta!" << endl;
        while (cin.get() != '\n');
        return;
    }

    imageClass imageData;

    if(image.channels() == 1){
        imageData.isRGB = false;
        cvtColor(image, image, COLOR_GRAY2RGB);
    }

    imageData.setImage(image);
    imageData.setRequisito(requisito);

    namedWindow("Image",WINDOW_AUTOSIZE);
    imshow("Image", image);
    setMouseCallback("Image", mouseClick, &imageData);
    waitKey(0);
    destroyAllWindows();
}

//Abre videoCapture para realizar os requisitos 1 e 2.
void video(int requisito){
	system("clear");
    VideoCapture capture;
    Mat frame;
    imageClass imageData;
    string windowName;

    if(requisito == REQUISITO3){
    	string videoName;
    	cout << "Escreva o nome do video que quer usar (incluindo a extenção):";
    	cin >> videoName;
    	while (cin.get() != '\n');
    	capture.open(videoName);
	    if(!capture.isOpened()){
	        cout << "Não foi possível abrir o video!" << endl;
	        while (cin.get() != '\n');
	        return;
	    }
	    windowName = "Video";
    }

    if(requisito == REQUISITO4){
    	capture.open(0);
	    if(!capture.isOpened()){
	        cout << "Não foi possível abrir a webcam!" << endl;
	        while (cin.get() != '\n');
	        return;
	    }
	    windowName = "Webcam";
    }


    while(1){
        capture >> frame;
        if(frame.empty()){
            break;
        }
        if(frame.channels() == 1){
            imageData.isRGB = false;
            cvtColor(frame, frame, COLOR_GRAY2RGB);
        }
        imageData.setImage(frame);
        imageData.setRequisito(requisito);

        namedWindow(windowName, WINDOW_AUTOSIZE);
        setMouseCallback(windowName, mouseClick, &imageData);

        if(!imageData.click){
            imshow(windowName, frame);
        }
        else{
            drawRedPixels(&imageData, windowName);
        }
        if (waitKey(30)>= 0){
        	break;
        }
    }

    destroyAllWindows();
}