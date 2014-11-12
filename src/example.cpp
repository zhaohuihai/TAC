/*
 * example.cpp
 *
 *  Created on: 2014��10��16��
 *      Author: zhaohuihai
 */

#include "example.h"

void tensor_contraction()
{
	Index x0(3), x1(2), x2(4), x3(5), x4(4);
	Tensor<double> A(x0, x1, x2, x3) ;
	Tensor<double> B(x1, x4, x2) ;
	A.randUniform() ;
	B.randUniform() ;

	Tensor<double> AB = contractTensors(A, B) ;
}


void tensor_reshape()
{
	Index x0(3), x1(2), x2(4);
	Tensor<double> A(x0, x1, x2) ;
	A.randUniform() ;

	A = A.reshape(Index(x0, x1), x2) ;
}

void tensor_permute()
{
	Index x0(3), x1(2), x2(4);
	Tensor<double> A(x0, x1, x2) ;
	A.randUniform() ;

	A = A.permute(x1, x0, x2) ;
}

void tensor_svd()
{
	int D = 3000 ;
	Index x1(D), x2(D) ;
//	std::cout << x1.dim() << std::endl ;
//	std::cout << x2.dim() << std::endl ;
	Tensor<double> A(x1, x2) ;

	A.randUniform() ;
	Tensor<double> U, S, V ;

	double time_start = dsecnd() ;

	svd(A, U, S, V) ;

	std::cout << std::endl << "Total CPU time = " << (dsecnd() - time_start ) << " seconds" << std::endl ;

}
