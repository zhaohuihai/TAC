/*
 * tensor_manipulation.h
 *
 *  Created on: 2011-5-25
 *      Author: zhaohuihai
 */

#ifndef tensor_manipulation_H_
#define tensor_manipulation_H_

#include <iostream>
#include <stdlib.h>
#include <valarray>

#include "tensor.h"

using namespace std ;

void printVector(Tensor &A) ;
void printMatrix(Tensor &A) ;
bool isAllDimAgree(Tensor &A, Tensor &B) ;
double norm(valarray<double> &V) ;

Tensor contractTensors(Tensor &A, int a1, Tensor &B, int b1) ;
Tensor contractTensors(Tensor &A, int a1, int a2, Tensor &B, int b1, int b2) ;
Tensor contractTensors(Tensor &A, int a1, int a2, int a3, Tensor &B, int b1, int b2, int b3) ;
Tensor contractTensors(Tensor &A, int a1, int a2, int a3, int a4, Tensor &B, int b1, int b2, int b3, int b4) ;
Tensor contractTensors(Tensor A, valarray<int> indA_right, Tensor B, valarray<int> indB_left) ;
valarray<int> findOuterIndex(int size, valarray<int> &inner_index) ;
void convertTensor2Matrix(Tensor &A, valarray<int> &indA_left, valarray<int> &indA_right) ;
Tensor computeMatrixProduct(Tensor &A, Tensor &B) ;
void convertMatrix2Tensor(Tensor &AB, valarray<int> &dimA, valarray<int> &indA_left, valarray<int> &dimB, valarray<int> &indB_right) ;
double * convertTensor2Array(Tensor &A) ;

// Computes all eigenvalues and, eigenvectors of a real generalized symmetric definite eigenproblem.
Tensor symGenEig(Tensor &A, Tensor &B, Tensor &U) ;
Tensor symGenEig(Tensor &A, Tensor &B, Tensor &U, int k) ;

// Computes eigenvalues and eigenvectors of a real symmetric matrix using the Relatively Robust Representations.
Tensor symEig(Tensor &A, Tensor &U) ;

// Upper triangular part of matrix
Tensor triu(Tensor A) ;
Tensor triu(Tensor A, int k) ;
Tensor trans(Tensor A) ;
// ====================================template functions===============================================
template <class T> T prod(valarray<T> &V)
{
	T n ;
	n = 1 ;
	for (int i = 0; i < V.size(); i ++)
	{
		n = n * V[i] ;
	}
	return n ;
}
// Product of elements of "V" at "index"
template <class T> T prod(valarray<T> &V, valarray<int> &index)
{
	T n ;
	n = 1 ;
	int i ;

	for (i = 0; i < index.size(); i ++)
	{
		if (index[i] >= V.size() || index[i] < 0)
		{
			cout << " Index exceeds matrix dimensions." << endl ;
			exit(0) ;
		}
		n = n * V[ index[i] ] ;
	}
	return n ;
}



#endif /* TENSORMANIPULATION_H_ */
