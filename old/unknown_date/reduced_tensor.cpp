/*
 * reduced_tensor.cpp
 *
 *  Created on: 2011-6-23
 *      Author: zhaohuihai
 */
#include <iostream>
#include <valarray>

#include "tensor.h"
#include "reduced_tensor.h"
#include "tensor_manipulation.h"

using namespace std ;

// public member functions
ReducedTensor::ReducedTensor() {}

ReducedTensor::ReducedTensor(WaveFunction &wave)
{
	// T1((x1,x1'),(y1,y1')) = sum{S1}_[A1(x1,y1,S1)*A1(x1',y1',S1)]
	// T1((x1,x1'),(y1,y1')) ~ T1(X1,Y1)
	T1 = createT1(wave.A1) ;

	// T2((x1,x1'),(x2,x2'),(y3,y3')) = sum{S2}_[A2(x1,x2,y3,S2)*A2(x1',x2',y3',S2)]
	// T2((x1,x1'),(x2,x2'),(y3,y3')) ~ T2(X1,X2,Y3)
	T2 = createT2(wave.A2) ;

	// T3((x2,x2'),(y5,y5')) = sum{S3}_[A3(x2,y5,S3)*A3(x2',y5',S3)]
	// T3((x2,x2'),(y5,y5')) ~ T3(X2,Y5)
	T3 = createT1(wave.A3) ;

	// T4((x3,x3'),(y1,y1'),(y2,y2')) = sum{S4}_[A4(x3,y1,y2,S4)*A4(x3',y1',y2',S4)]
	// T4((x3,x3'),(y1,y1'),(y2,y2')) ~ T4(X3,Y1,Y2)
	T4 = createT2(wave.A4)  ;

	// T5((x3,x3'),(x4,x4'),(y3,y3'),(y4,y4')) = sum{S5}_[A5(x3,x4,y3,y4,S5)*A5(x3',x4',y3',y4',S5)]
	// T5((x3,x3'),(x4,x4'),(y3,y3'),(y4,y4')) ~ T5(X3,X4,Y3,Y4)
	T5 = createT5(wave.A5) ;

	// T6((x4,x4'),(y5,y5'),(y6,y6')) = sum{S6}_[A6(x4,y5,y6,S6)*A6(x4',y5',y6',S6)]
	// T6((x4,x4'),(y5,y5'),(y6,y6')) ~ T6(X4,Y5,Y6)
	T6 = createT2(wave.A6) ;

	// T7((x5,x5'),(y2,y2')) = sum{S7}_[A7(x5,y2,S7)*A7(x5',y2',S7)]
	// T7((x5,x5'),(y2,y2')) ~ T7(X5,Y2)
	T7 = createT1(wave.A7) ;

	// T8((x5,x5'),(x6,x6'),(y4,y4')) = sum{S8}_[A8(x5,x6,y4,S8)*A8(x5',x6',y4',S8)]
	// T8((x5,x5'),(x6,x6'),(y4,y4')) ~ T8(X5,X6,Y4)
	T8 = createT2(wave.A8) ;

	// T9((x6,x6'),(y6,y6')) = sum{S9}_[A9(x6,y6,S9)*A9(x6',y6',S9)]
	// T9((x6,x6'),(y6,y6')) ~ T9(X6,Y6)
	T9 = createT1(wave.A9) ;

}

ReducedTensor::~ReducedTensor() {}
//***********************************manipulation******************************************************

/*                          rotate counter-clockwise
 *
 *   T1---(X1)---T2---(X2)---T3              T3---(Y5)---T6---(Y6)---T9
 *   |           |           |               |           |           |
 *   |           |           |				 |           |           |
 *  (Y1)   		(Y3)        (Y5)			(X2)   		(X4)        (X6)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   T4---(X3)---T5---(X4)---T6       ==>    T2---(Y3)---T5---(Y4)---T8
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *  (Y2)        (Y4)        (Y6)            (X1)        (X3)        (X5)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   T7---(X5)---T8---(X6)---T9              T1---(Y1)---T4---(Y2)---T7
 */
void ReducedTensor::rotate()
{
	ReducedTensor rt0 = *this ;
	// T3(X2,Y5) => T1(Y1,X1)
	T1 = rt0.T3 ;
	// T1(Y1,X1) => T1(X1,Y1)
	T1.permute(2, 1) ;

	// T6(X4,Y5,Y6) => T2(Y3,X1,X2)
	T2 = rt0.T6 ;
	// T2(Y3,X1,X2) => T2(X1,X2,Y3)
	T2.permute(2, 3, 1) ;

	// T9(X6,Y6) => T3(Y5,X2)
	T3 = rt0.T9 ;
	// T3(Y5,X2) => T3(X2,Y5)
	T3.permute(2, 1) ;

	// T2(X1,X2,Y3) => T4(Y2,Y1,X3)
	T4 = rt0.T2 ;
	// T4(Y2,Y1,X3) => T4(X3,Y1,Y2)
	T4.permute(3, 2, 1) ;

	// T5(X3,X4,Y3,Y4) => T5(Y4,Y3,X3,X4)
	T5 = rt0.T5 ;
	// T5(Y4,Y3,X3,X4) => T5(X3,X4,Y3,Y4)
	T5.permute(3, 4, 2, 1) ;

	// T8(X5,X6,Y4) => T6(Y6,Y5,X4)
	T6 = rt0.T8 ;
	// T6(Y6,Y5,X4) => T6(X4,Y5,Y6)
	T6.permute(3, 2, 1) ;

	// T1(X1,Y1) => T7(Y2,X5)
	T7 = rt0.T1 ;
	// T7(Y2,X5) => T7(X5,Y2)
	T7.permute(2, 1) ;

	// T4(X3,Y1,Y2) => T8(Y4,X5,X6)
	T8 = rt0.T4 ;
	// T8(Y4,X5,X6) => T8(X5,X6,Y4)
	T8.permute(2, 3, 1) ;

	// T7(X5,Y2) => T9(Y6,X6)
	T9 = rt0.T7 ;
	// T9(Y6,X6) => T9(X6,Y6)
	T9.permute(2, 1) ;
}
/*
 *   T1---(X1)---T2---(X2)---T3              T1---(Y1)---T4---(Y2)---T7
 *   |           |           |               |           |           |
 *   |           |           |				 |           |           |
 *  (Y1)   		(Y3)        (Y5)			(X1)   		(X3)        (X5)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   T4---(X3)---T5---(X4)---T6       ==>    T2---(Y3)---T5---(Y4)---T8
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *  (Y2)        (Y4)        (Y6)            (X2)        (X4)        (X6)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   T7---(X5)---T8---(X6)---T9              T3---(Y5)---T6---(Y6)---T9
 */
void ReducedTensor::reflectAxis19() // (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
{
	ReducedTensor rt0 = *this ;
	// T1(X1,Y1) => T1(Y1,X1)
	T1 = rt0.T1 ;
	// T1(Y1,X1) => T1(X1,Y1)
	T1.permute(2, 1) ;

	// T4(X3,Y1,Y2) => T2(Y3,X1,X2)
	T2 = rt0.T4 ;
	// T2(Y3,X1,X2) => T2(X1,X2,Y3)
	T2.permute(2, 3, 1) ;

	// T7(X5,Y2) => T3(Y5,X2)
	T3 = rt0.T7 ;
	// T3(Y5,X2) => T3(X2,Y5)
	T3.permute(2, 1) ;

	// T2(X1,X2,Y3) => T4(Y1,Y2,X3)
	T4 = rt0.T2 ;
	// T4(Y1,Y2,X3) => T4(X3,Y1,Y2)
	T4.permute(3, 1, 2) ;

	// T5(X3,X4,Y3,Y4) => T5(Y3,Y4,X3,X4)
	T5 = rt0.T5 ;
	// T5(Y3,Y4,X3,X4) => T5(X3,X4,Y3,Y4)
	T5.permute(3, 4, 1, 2) ;

	// T8(X5,X6,Y4) => T6(Y5,Y6,X4)
	T6 = rt0.T8 ;
	// T6(Y5,Y6,X4) => T6(X4,Y5,Y6)
	T6.permute(3, 1, 2) ;

	// T3(X2,Y5) => T7(Y2,X5)
	T7 = rt0.T3 ;
	// T7(Y2,X5) => T7(X5,Y2)
	T7.permute(2, 1) ;

	// T6(X4,Y5,Y6) => T8(Y4,X5,X6)
	T8 = rt0.T6 ;
	// T8(Y4,X5,X6) => T8(X5,X6,Y4)
	T8.permute(2, 3, 1) ;

	// T9(X6,Y6) => T9(Y6,X6)
	T9 = rt0.T9 ;
	// T9(Y6,X6) => T9(X6,Y6)
	T9.permute(2, 1) ;
}
/*
 *   T1---(X1)---T2---(X2)---T3              T3---(X2)---T2---(X1)---T1
 *   |           |           |               |           |           |
 *   |           |           |				 |           |           |
 *  (Y1)   		(Y3)        (Y5)			(Y5)   		(Y3)        (Y1)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   T4---(X3)---T5---(X4)---T6       ==>    T6---(X4)---T5---(X3)---T4
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *  (Y2)        (Y4)        (Y6)            (Y6)        (Y4)        (Y2)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   T7---(X5)---T8---(X6)---T9              T9---(X6)---T8---(X5)---T7
 */
void ReducedTensor::reflectAxis28() // (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
{
	ReducedTensor rt0 = *this ;
	// T3(X2,Y5) => T1(X1,Y1)
	T1 = rt0.T3 ;

	// T2(X1,X2,Y3) => T2(X2,X1,Y3)
	T2 = rt0.T2 ;
	// T2(X2,X1,Y3) => T2(X1,X2,Y3)
	T2.permute(2, 1, 3) ;

	// T1(X1,Y1) => T3(X2,Y5)
	T3 = rt0.T1 ;

	// T6(X4,Y5,Y6) => T4(X3,Y1,Y2)
	T4 = rt0.T6 ;

	// T5(X3,X4,Y3,Y4) => T5(X4,X3,Y3,Y4)
	T5 = rt0.T5 ;
	// T5(X4,X3,Y3,Y4) => T5(X3,X4,Y3,Y4)
	T5.permute(2, 1, 3, 4) ;

	// T4(X3,Y1,Y2) => T6(X4,Y5,Y6)
	T6 = rt0.T4 ;

	// T9(X6,Y6) => T7(X5,Y2)
	T7 = rt0.T9 ;

	// T8(X5,X6,Y4) => T8(X6,X5,Y4)
	T8 = rt0.T8 ;
	// T8(X6,X5,Y4) => T8(X5,X6,Y4)
	T8.permute(2, 1, 3) ;

	// T7(X5,Y2) => T9(X6,Y6)
	T9 = rt0.T7 ;
}
// ======================================private member functions======================================
Tensor ReducedTensor::createT1(Tensor &A)
{
	// indA = [3] ;
	valarray<int> indA(3, 1) ;

	// T(x1,y1,x1',y1') = sum{S1}_[A(x1,y1,S1)*A(x1',y1',S1)]
	Tensor T = contractTensors(A, indA, A, indA) ;
	//Tensor T = contractTensors(A, 3, A, 3) ;
	//exit(0) ;
	int r [] = {1, 3, 2, 4} ;
	valarray<int> order(r, 4) ;
	// T(x1,y1,x1',y1') => T(x1,x1',y1,y1')
	T.permute(order) ;

	valarray<int> T_dim(1, 2) ;
	int i ;
	for (i = 0; i < 2; i ++)
	{
		T_dim[i] = A.dim[i] * A.dim[i] ;
	}
	// T((x1,x1'),(y1,y1')) ~ T(X1,Y1)
	T.dim = T_dim ;

	return T ;
}

Tensor ReducedTensor::createT2(Tensor &A)
{
	// indA = [4] ;
	valarray<int> indA(4, 1) ;

	// T(x1,x2,y3,x1',x2',y3') = sum{S2}_[A(x1,x2,y3,S2)*A(x1',x2',y3',S2)]
	Tensor T = contractTensors(A, indA, A, indA) ;
	int r [] = {1, 4, 2, 5, 3, 6} ;
	valarray<int> order(r, 6) ;
	// T(x1,x2,y3,x1',x2',y3') => T(x1,x1',x2,x2',y3,y3')
	T.permute(order) ;

	valarray<int> T_dim(1, 3) ;
	int i ;
	for (i = 0; i < 3; i ++)
	{
		T_dim[i] = A.dim[i] * A.dim[i] ;
	}
	// T((x1,x1'),(x2,x2'),(y3,y3')) ~ T2(X1,X2,Y3)
	T.dim = T_dim ;

	return T ;
}

Tensor ReducedTensor::createT5(Tensor &A)
{
	// indA = [5] ;
	valarray<int> indA(5, 1) ;

	// T(x3,x4,y3,y4,x3',x4',y3',y4') = sum{S5}_[A(x3,x4,y3,y4,S5)*A(x3',x4',y3',y4',S5)]
	Tensor T = contractTensors(A, indA, A, indA) ;
	int r [] = {1, 5, 2, 6, 3, 7, 4, 8} ;
	valarray<int> order(r, 8) ;
	// T(x3,x4,y3,y4,x3',x4',y3',y4') => T(x3,x3',x4,x4',y3,y3',y4,y4')
	T.permute(order) ;

	valarray<int> T_dim(1, 4) ;
	int i ;
	for (i = 0; i < 4; i ++)
	{
		T_dim[i] = A.dim[i] * A.dim[i] ;
	}
	// T((x3,x3'),(x4,x4'),(y3,y3'),(y4,y4')) ~ T(X3,X4,Y3,Y4)
	T.dim = T_dim ;

	return T ;
}
