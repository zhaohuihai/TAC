/*
 * parameter.cpp
 *
 *  Created on: 2011-6-9
 *      Author: zhaohuihai
 */
#include "parameter.h"
#include <valarray>
#include "tensor.h"

using namespace std ;

Parameter_eigifp::Parameter_eigifp()
{
	guessVec = 0 ;
	tolerance = Default ;
	maxIteration = 0 ;
	innerIteration = 0 ;
	disp = 0 ;
}



Parameter::Parameter()
{
	// Bond dimension
	D1 = 4 ;
	D2 = 4 ;
	// site dimension
	d0 = 2 ;
	// max block dimension
	d1 = 30 ;
	d2 = 30 ;
	// final lattice size
	lengthFinal = 12 ;
	widthFinal = 12 ;
	// variation convergence tolerance
	tol_variation = 1e-6 ;
	// Maximum number of iterations in variation
	maxIter_variation = 1e3 ;

	// display intermediate result
	disp = 1 ;
}

void Parameter::rotate()
{
	Parameter parameter0 = *this ;
	lengthFinal = parameter0.widthFinal ;
	widthFinal = parameter0.lengthFinal ;
}


Parameter::~Parameter() { }
