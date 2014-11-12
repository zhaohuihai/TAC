/*
 * wave_function.h
 *
 *  Created on: 2011-6-21
 *      Author: zhaohuihai
 */

#ifndef WAVE_FUNCTION_H_
#define WAVE_FUNCTION_H_

#include "tensor.h"
#include "parameter.h"
#include "hamiltonian.h"

class WaveFunction
{
public:
	// constructor
	WaveFunction() ;
	WaveFunction(Parameter &parameter, Hamiltonian &hamiltonian) ;
	// destructor
	~WaveFunction() ;

	Tensor A1, A2, A3, A4, A5, A6, A7, A8, A9 ;
	//*********************manipulation******************************************************
	void rotate() ; // (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A6,A9,A2,A5,A8,A1,A4,A7)
	void reflectAxis19() ; // (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	void reflectAxis28() ; // (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
};

#endif /* WAVE_FUNCTION_H_ */
