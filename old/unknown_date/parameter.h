/*
 * parameter.h
 *
 *  Created on: 2011-5-31
 *      Author: zhaohuihai
 */

#ifndef PARAMETER_H_
#define PARAMETER_H_

#include "tensor.h"

#define Default        0
#define EPS            2.22044604925031e-16

class Parameter_eigifp
{
public:
	Parameter_eigifp() ;

	bool guessVec ;
	double tolerance ;
	int innerIteration ;
	int maxIteration ;
	bool disp ;
};

class Parameter
{
public:
	//constructor
	Parameter() ;
	//destructor
	~Parameter() ;
	//Bond dimension
	int D1, D2 ;
	// site dimension
	int d0 ;
	// max block dimension
	int d1 ;
	int d2 ;
	// final lattice size
	int lengthFinal ;
	int widthFinal ;
	// variation convergence tolerance
	double tol_variation ;
	// Maximum number of iterations in variation
	int maxIter_variation ;

	bool disp ;

	Parameter_eigifp eigifp_opt ;
	// rotate the lattice by 90 degree
	void rotate() ;
};


#endif /* PARAMETER_H_ */
