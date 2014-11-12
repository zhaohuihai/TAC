/*
 * wave_function.cpp
 *
 *  Created on: 2011-6-21
 *      Author: zhaohuihai
 */
#include <iostream>

#include "tensor.h"
#include "parameter.h"
#include "hamiltonian.h"
#include "wave_function.h"

using namespace std ;

WaveFunction::WaveFunction() {}
WaveFunction::WaveFunction(Parameter &parameter, Hamiltonian &hamiltonian)
{
	int D1 = parameter.D1 ;
	int D2 = parameter.D2 ;
	int d0 = parameter.d0 ;

	int B1 = hamiltonian.B1 ;
	int  a1 [] = {D1, D1, B1} ;
	valarray<int> dim1(a1, 3) ;
	// A1(x1,y1,S1)
	A1 = Tensor(dim1, RandUniform) ;

	int B2 = hamiltonian.B2 ;
	int a2 [] = {D1, D1, D2, B2} ;
	valarray<int> dim2(a2, 4) ;
	// A2(x1,x2,y3,S2)
	A2 = Tensor(dim2, RandUniform) ;

	int B3 = hamiltonian.B3 ;
	int a3 [] = {D1, D1, B3} ;
	valarray<int> dim3(a3, 3) ;
	// A3(x2,y5,S3)
	A3 = Tensor(dim3, RandUniform) ;

	int B4 = hamiltonian.B4 ;
	int a4 [] = {D2, D1, D1, B4} ;
	valarray<int> dim4(a4, 4) ;
	// A4(x3,y1,y2,S4)
	A4 = Tensor(dim4, RandUniform) ;

	int a5 [] = {D2, D2, D2, D2, d0} ;
	valarray<int> dim5(a5, 5) ;
	// A5(x3,x4,y3,y4,S5)
	A5 = Tensor(dim5, RandUniform) ;

	int B6 = hamiltonian.B6 ;
	int a6 [] = {D2, D1, D1, B6} ;
	valarray<int> dim6(a6, 4) ;
	// A6(x4,y5,y6,S6)
	A6 = Tensor(dim6, RandUniform) ;

	int B7 = hamiltonian.B7 ;
	int a7 [] = {D1, D1, B7} ;
	valarray<int> dim7(a7, 3) ;
	// A7(x5,y2,S7)
	A7 = Tensor(dim7, RandUniform) ;

	int B8 = hamiltonian.B8 ;
	int a8 [] = {D1, D1, D2, B8} ;
	valarray<int> dim8(a8, 4) ;
	// A8(x5,x6,y4,S8)
	A8 = Tensor(dim8, RandUniform) ;

	int B9 = hamiltonian.B9 ;
	int a9 [] = {D1, D1, B9} ;
	valarray<int> dim9(a9, 3) ;
	// A9(x6,y6,S9)
	A9 = Tensor(dim9, RandUniform) ;
}
/*
 *   A1---(x1)---A2---(x2)---A3              A3---(y5)---A6---(y6)---A9
 *   |           |           |               |           |           |
 *   |           |           |				 |           |           |
 *  (y1)   		(y3)        (y5)			(x2)   		(x4)        (x6)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   A4---(x3)---A5---(x4)---A6       ==>    A2---(y3)---A5---(y4)---A8
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *  (y2)        (y4)        (y6)            (x1)        (x3)        (x5)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   A7---(x5)---A8---(x6)---A9              A1---(y1)---A4---(y2)---A7
 */
void WaveFunction::rotate() // rotate counter-clockwise
{
	WaveFunction wave0 = *this ;
	// A3(x2,y5,S3) => A1(y1,x1,S1)
	A1 = wave0.A3 ;
	// A1(y1,x1,S1) => A1(x1,y1,S1)
	int r1 [] = {2, 1, 3} ;
	valarray<int> order1(r1, 3) ;
	A1.permute(order1) ;

	// A6(x4,y5,y6,S6) => A2(y3,x1,x2,S2)
	A2 = wave0.A6 ;
	// A2(y3,x1,x2,S2) => A2(x1,x2,y3,S2)
	int r2 [] = {2, 3, 1, 4} ;
	valarray<int> order2(r2, 4) ;
	A2.permute(order2) ;

	// A9(x6,y6,S9) => A3(y5,x2,S3)
	A3 = wave0.A9 ;
	// A3(y5,x2,S3) => A3(x2,y5,S3)
	A3.permute(order1) ;

	// A2(x1,x2,y3,S2) => A4(y2,y1,x3,S4)
	A4 = wave0.A2 ;
	// A4(y2,y1,x3,S4) => A4(x3,y1,y2,S4)
	int r4 [] = {3, 2, 1, 4} ;
	valarray<int> order4(r4, 4) ;
	A4.permute(order4) ;

	// A5(x3,x4,y3,y4,S5) => A5(y4,y3,x3,x4,S5)
	A5 = wave0.A5 ;
	// A5(y4,y3,x3,x4,S5) => A5(x3,x4,y3,y4,S5)
	int r5 [] = {3, 4, 2, 1, 5} ;
	valarray<int> order5(r5, 5) ;
	A5.permute(order5) ;

	// A8(x5,x6,y4,S8) => A6(y6,y5,x4,S6)
	A6 = wave0.A8 ;
	// A6(y6,y5,x4,S6) => A6(x4,y5,y6,S6)
	A6.permute(order4) ;

	// A1(x1,y1,S1) => A7(y2,x5,S7)
	A7 = wave0.A1 ;
	// A7(y2,x5,S7) => A7(x5,y2,S7)
	A7.permute(order1) ;

	// A4(x3,y1,y2,S4) => A8(y4,x5,x6,S8)
	A8 = wave0.A4 ;
	// A8(y4,x5,x6,S8) => A8(x5,x6,y4,S8) [2,3,1,4]
	A8.permute(order2) ;

	// A7(x5,y2,S7) => A9(y6,x6,S9)
	A9 = wave0.A7 ;
	// A9(y6,x6,S9) => A9(x6,y6,S9) [2,1,3]
	A9.permute(order1) ;
}
/*
 *   A1---(x1)---A2---(x2)---A3              A1---(y1)---A4---(y2)---A7
 *   |           |           |               |           |           |
 *   |           |           |				 |           |           |
 *  (y1)   		(y3)        (y5)			(x1)   		(x3)        (x5)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   A4---(x3)---A5---(x4)---A6       ==>    A2---(y3)---A5---(y4)---A8
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *  (y2)        (y4)        (y6)            (x2)        (x4)        (x6)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   A7---(x5)---A8---(x6)---A9              A3---(y5)---A6---(y6)---A9
 */
void WaveFunction::reflectAxis19()
{
	WaveFunction wave0 = *this ;
	// A1(x1,y1,S1) => A1(y1,x1,S1)
	A1 = wave0.A1 ;
	// A1(y1,x1,S1) => A1(x1,y1,S1) [2, 1, 3]
	A1.permute(2, 1, 3) ;

	// A4(x3,y1,y2,S4) => A2(y3,x1,x2,S2)
	A2 = wave0.A4 ;
	// A2(y3,x1,x2,S2) => A2(x1,x2,y3,S2)
	A2.permute(2, 3, 1, 4) ;

	// A7(x5,y2,S7) => A3(y5,x2,S3)
	A3 = wave0.A7 ;
	// A3(y5,x2,S3) => A3(x2,y5,S3)
	A3.permute(2, 1, 3) ;

	// A2(x1,x2,y3,S2) => A4(y1,y2,x3,S4)
	A4 = wave0.A2 ;
	// A4(y1,y2,x3,S4) => A4(x3,y1,y2,S4)
	A4.permute(3, 1, 2, 4) ;

	// A5(x3,x4,y3,y4,S5) => A5(y3,y4,x3,x4,S5)
	A5 = wave0.A5 ;
	// A5(y3,y4,x3,x4,S5) => A5(x3,x4,y3,y4,S5)
	A5.permute(3, 4, 1, 2, 5) ;

	// A8(x5,x6,y4,S8) => A6(y5,y6,x4,S6)
	A6 = wave0.A8 ;
	// A6(y5,y6,x4,S6) => A6(x4,y5,y6,S6)
	A6.permute(3, 1, 2, 4) ;

	// A3(x2,y5,S3) => A7(y2,x5,S7)
	A7 = wave0.A3 ;
	// A7(y2,x5,S7) => A7(x5,y2,S7)
	A7.permute(2, 1, 3) ;

	// A6(x4,y5,y6,S6) => A8(y4,x5,x6,S8)
	A8 = wave0.A6 ;
	// A8(y4,x5,x6,S8) => A8(x5,x6,y4,S8)
	A8.permute(2, 3, 1, 4) ;

	// A9(x6,y6,S9) => A9(y6,x6,S9)
	A9 = wave0.A9 ;
	// A9(y6,x6,S9) => A9(x6,y6,S9)
	A9.permute(2, 1, 3) ;
}
/*
 *   A1---(x1)---A2---(x2)---A3              A3---(x2)---A2---(x1)---A1
 *   |           |           |               |           |           |
 *   |           |           |				 |           |           |
 *  (y1)   		(y3)        (y5)			(y5)   		(y3)        (y1)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   A4---(x3)---A5---(x4)---A6       ==>    A6---(x4)---A5---(x3)---A4
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *  (y2)        (y4)        (y6)            (y6)        (y4)        (y2)
 *   |           |           |               |           |           |
 *   |           |           |               |           |           |
 *   A7---(x5)---A8---(x6)---A9              A9---(x6)---A8---(x5)---A7
 */
void WaveFunction::reflectAxis28()
{
	WaveFunction wave0 = *this ;
	// A3(x2,y5,S3) => A1(x1,y1,S1)
	A1 = wave0.A3 ;

	// A2(x1,x2,y3,S2) => A2(x2,x1,y3,S2)
	A2 = wave0.A2 ;
	// A2(x2,x1,y3,S2) => A2(x1,x2,y3,S2)
	A2.permute(2, 1, 3, 4) ;

	// A1(x1,y1,S1) => A3(x2,y5,S3)
	A3 = wave0.A1 ;

	// A6(x4,y5,y6,S6) => A4(x3,y1,y2,S4)
	A4 = wave0.A6 ;

	// A5(x3,x4,y3,y4,S5) => A5(x4,x3,y3,y4,S5)
	A5 = wave0.A5 ;
	// A5(x4,x3,y3,y4,S5) => A5(x3,x4,y3,y4,S5)
	A5.permute(2, 1, 3, 4, 5) ;

	// A4(x3,y1,y2,S4) => A6(x4,y5,y6,S6)
	A6 = wave0.A4 ;

	// A9(x6,y6,S9) => A7(x5,y2,S7)
	A7 = wave0.A9 ;

	// A8(x5,x6,y4,S8) => A8(x6,x5,y4,S8)
	A8 = wave0.A8 ;
	// A8(x6,x5,y4,S8) => A8(x5,x6,y4,S8)
	A8.permute(2, 1, 3, 4) ;

	// A7(x5,y2,S7) => A9(x6,y6,S9)
	A9 = wave0.A7 ;
}

WaveFunction::~WaveFunction() {}
