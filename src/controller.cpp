/*  Aluna: Natalia Oliveira Borges
    Matricula: 16/0015863
    Projeto Demonstrativo 1 - Explorando OpenCV 
*/

#include "controller.hpp"

Controller::Controller(){
	clicks = 0;

	distcoef.push_back(0);
	distcoef.push_back(0);
	distcoef.push_back(0);
	distcoef.push_back(0);
	distcoef.push_back(0);
}

Controller::~Controller(){}

