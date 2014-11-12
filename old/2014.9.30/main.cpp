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

int main(void)
{
//	 start measuring time cost

//	int D = 2000 ;
//	Tensor<double> A(4, 3) ;
//	A.arithmeticSequence(0, 2) ;
//	Tensor<double> B = A + 1.0 ;
////	B.arithmeticSequence(1, 2) ;
//
//	Tensor<int> P(3) ;
//	P[0] = 2 ;
//	P[1] = 0 ;
//	P[2] = 1 ;

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

	Index x0, x1, x2, x3, x4, x5 ;
	Tensor<double> A(2, 3, 4, 5) ;
	A.arithmeticSequence(0, 1) ;
	Tensor<double> B(2, 4, 3) ;
	B.arithmeticSequence(0, 2) ;

	A.putIndex(x1, x2, x4, x5) ;
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

