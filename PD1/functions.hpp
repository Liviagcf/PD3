/*  Aluna: Natalia Oliveira Borges
    Matricula: 16/0015863
    Projeto Demonstrativo 1 - Explorando OpenCV 
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "imageClass.hpp"

using namespace std;
using namespace cv; 

#define REQUISITO1 1 
#define REQUISITO2 2 
#define REQUISITO3 3 
#define REQUISITO4 4


//getPixelValues é responsável por imprimir na tela os valores RGB ou grayscale do pixel clicado.
void getPixelValues(int row, int col, void* userdata);

//drawRedPixels é responsável por pintar os pixels 13 tons próximos do pixel clicado de vermelho.
void drawRedPixels(void* userdata, string windowName);

//mouseClick é responsável por identificar as coordenadas do pixel desejado. 
void mouseClick(int event, int x, int y, int flags, void* userdata);

//Abre imagem para realizar os requisitos 1 e 2.
void image(int requisito);

//Abre videoCapture para realizar os requisitos 1 e 2.
void video(int requisito);