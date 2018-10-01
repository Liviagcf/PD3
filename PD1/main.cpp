/*  Aluna: Natalia Oliveira Borges
    Matricula: 16/0015863
    Projeto Demonstrativo 1 - Explorando OpenCV 
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "functions.hpp"

using namespace std;
using namespace cv;



void menu(){
    string choice;
    bool invalid = false;

    while(1){

        system("clear");

        cout << "Qual requisito deseja testar:" << endl << endl << endl;
        cout << "1) Requisito 1 " << endl;
        cout << "    Abre uma imagem e, ao clicar em um pixel, mostra sua linha, coluna e intensidade do pixel em grayscale ou RGB." << endl << endl;
        cout << "2) Requisito 2 " << endl;
        cout << "    Faz o Requisito 1 e colore o pixel escolhido e os pixels 13 tons proximos dele de vermelho." << endl << endl;
        cout << "3) Requisito 3 " << endl;
        cout << "    Abre um video e realiza o Requisito 2." << endl << endl;
        cout << "4) Requisito 4 " << endl;
        cout << "   Abre a webcam e realiza o Requisito 2." << endl << endl;
        cout << "5) Sair" << endl << endl;

        if(invalid){
            cout << "Nao tem essa opção!" << endl << endl;
            invalid = false;
        }

        cout << "Sua escolha:" << endl;

        cin >> choice;
        while (cin.get() != '\n');


        if(choice == "1"){
            image(REQUISITO1);
        }
        else if(choice == "2"){
            image(REQUISITO2);
        }
        else if(choice == "3"){
            video(3);
        }
        else if(choice == "4"){
            video(4);
        }
        else if(choice == "5"){
            break;
        }
        else{
            invalid = true;
        }
    }
}

int main(int argc, char** argv){
    menu();
    return 0;
}