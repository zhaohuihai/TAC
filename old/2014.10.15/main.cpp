// google c++ style: http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml

//#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#include <mkl.h>

//#include <boost/shared_ptr.hpp>

#include "TensorOperation/TensorOperation.hpp"

Index test(Index& x1)
{
	return x1 ;
//	Index x2 = x1 ;
//	for (int i = 0 ; i < Index::inds_count().size(); i ++)
//	{
//		std::cout << Index::inds_count()[i] << ", " ;
//	}
//	std::cout << std::endl ;
//	return x2 ;
}

void test2(const Index& y)
{
	std::cout << " ind " << y.ind() << std::endl ;
	std::cout << " dim " << y.dim() << std::endl ;
}

int main(void)
{

	double time_start = dsecnd() ;

//	for (int j = 0; j < 100; j ++)
//	{
//		for (int i = 0; i < 1; i ++)
//		{
////			A = spitVector(A, 1, B) ;
//		}
//	}
//
//	TensorArray<double> T(2) ;
//	T(0) = A ;
//	T(1) = B ;

//	for (int i = 0 ; i < Index::all_ind().size(); i ++)
//	{
//		std::cout << Index::all_ind()[i] << ", " ;
//	}
//	std::cout << std::endl ;

	Index x0(2), x1(3), x2(2), x3(4), x4(3), x5(2), x6(2), x7(2) ;

	Tensor<double> A(x0, x1, x2, x3, x4) ;
	A.arithmeticSequence(0, 2) ;
	A.display1D() ;

	Tensor<double> A1 = A.permute(x1, x3, x2, x4, x0) ;
	A1.display1D() ;
	Tensor<double> A2 = A.permute( 1,  3,  2,  4,  0) ;
	A2.display1D() ;

	Tensor<double> AA = A1 - A2 ;
	std::cout << AA.maxAbs() << std::endl ;

	Tensor<double> A3 = A1.reshape(Index(x1,x3, x2), Index(x4, x0)) ;
	A3.display() ;

	exit(0) ;
//	Tensor<double> A(x0, x1, x2) ;
	A.arithmeticSequence(0, 1) ;
	Tensor<double> B(2, 4, 3) ;
	B.arithmeticSequence(0, 2) ;

	A.putIndex(x0, x2, x4) ;
	A.info() ;
	B.putIndex(x0, x4, x2) ;
	B.info() ;
	Tensor<double> AB = contractTensors(A, B) ;
	AB.info() ;
	AB.display1D() ;

	for (int i = 0 ; i < Index::inds_count().size(); i ++)
	{
		std::cout << Index::inds_count()[i] << ", " ;
	}
	std::cout << std::endl ;

	std::cout << std::endl << "Total CPU time = " << (dsecnd() - time_start ) << " seconds" << std::endl ;
  return 1;
}

