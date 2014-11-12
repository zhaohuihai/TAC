/*
 * reduced_tensor.h
 *
 *  Created on: 2011-6-23
 *      Author: zhaohuihai
 */

#ifndef REDUCED_TENSOR_H_
#define REDUCED_TENSOR_H_

#include "tensor.h"
#include "wave_function.h"


class ReducedTensor
{
public:
	// constructor
	ReducedTensor() ;
	ReducedTensor(WaveFunction &wave) ;
	// destructor
	~ReducedTensor() ;

	Tensor T1, T2, T3, T4, T5, T6, T7, T8, T9 ;
	//*********************manipulation******************************************************
	void rotate() ; // (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T6,T9,T2,T5,T8,T1,T4,T7)
	void reflectAxis19() ; // (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	void reflectAxis28() ; // (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
private:
	// T1 T3 T7 T9
	Tensor createT1(Tensor &A1) ;
	// T2 T4 T6 T8
	Tensor createT2(Tensor &A2) ;
	// T5
	Tensor createT5(Tensor &A5) ;
};

#endif /* REDUCED_TENSOR_H_ */
