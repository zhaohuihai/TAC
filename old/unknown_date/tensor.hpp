/*
 * Tensor.hpp
 *
 *  Created on: 2013-7-16
 *      Author: ZhaoHuihai
 */

#ifndef TENSOR_HPP_
#define TENSOR_HPP_

// C++ Standard Template Library
//	1. C library
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cassert>
//	2. Containers
#include <vector>
//	3. Input/Output
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
//	4. Other
#include <algorithm>
#include <string>
#include <complex>
#include <utility>
//-----------------------------
// intel mkl
#include <mkl.h>
//#include "mkl_lapacke.h"
//************************************************************************************
// do NOT use "using namespace"!

// Type
typedef std::complex<double> double_complex;

/*
 *  @class Tensor
 */
template <typename C>
class Tensor
{
private:

	// Rank of tensor
	int R ;
	/// Total number of elements
	int N ;
	/// Array of dimension of R indices
	int* D ;
	/// array of Values of N elements
	C* V ;
	//--------------------------------------------------------------------------------
	void printMatrix() ;
	C& at(Tensor<int> &position) ;
	bool outOfBound(Tensor<int> &position) ;
	void copy(const Tensor<C>& T) ;

public:
    // constructor
    Tensor() ; // empty tensor
    Tensor(const Tensor<C>& T) ; // overload copy constructor
    // create a d0 dimensional column vector, which is represented as a d0 X 1 matrix.
    Tensor(int d0) ; // all elements are 0
    Tensor(int d0, int d1) ;
    Tensor(int d0, int d1, int d2) ;
    Tensor(const std::vector<int>& dim) ;
    Tensor(const Tensor<int>& dim, C x) ;
    //     destructor
    	~Tensor() ;
    	//--------------
    	void randUniform() ;
    	void arithmeticSequence(C inital, C difference) ;
    	//---------------------------------------------------
    	// overload operators
    	Tensor<C>& operator = (const C a) ;
    	Tensor<C>& operator = (Tensor<C> T) ; // copy tensor
    	C& operator [] (int n) const ; // get an element by array index
    	// Access operator of an element in Fortran order. Get an element by tensor index
    	C& operator () (const int i1) ;
    	C& operator () (const int i1, const int i2) ;
    	C& operator () (const int i1, const int i2, const int i3) ;
    	C& operator () (Tensor<int> ind) ;

    	//---------------------------------------------------
    	Tensor<C> trans() ; // transpose matrix or vector
    	Tensor<C> ctrans() ;

    	Tensor<C> permute(int i1, int i2) ;
    	Tensor<C> permute(int i1, int i2, int i3) ;
    	Tensor<C> permute(Tensor<int> order) ;

    	Tensor<C> reshape(int i1) ;
    	Tensor<C> reshape(int i1, int i2) ;
    	Tensor<C> reshape(int i1, int i2, int i3) ;
    	Tensor<C> reshape(Tensor<int> dim) ;
    	//---------------------------------------------------
    	Tensor<int> arrayPosToTensorInd(int p) ; // convert array position to tensor index
    int tensorIndToArrayPos(Tensor<int> ind) ; // convert tensor index to array position
    Tensor<C> subTensor(int start, int end) ;
    Tensor<C> subTensor(int start1, int end1, int start2, int end2) ;
    Tensor<C> subTensor(Tensor<int>& start, Tensor<int>& end) ;

    //Tensor<C> diag(Tensor<C> T) ;
    void squeeze() ;
    //---------------------------------------------------
    C min() ;
    C max() ;
    C sum() ;
    C prod() const ;
    double norm(char normType) ;

    bool isSym() ;
    	//---------------------------------------------------
    	int rank() const ;
    	int numel() const ;
    	int* dimArr() const ;
    	C* ValArr() const ;
    	int dimension(int i) const ;

    	void display() ;
};
// non-member functions

// overload operators
template <typename C> Tensor<C> operator + (Tensor<C>& A, Tensor<C>& B) ;
template <typename C> Tensor<C> operator + (C a, Tensor<C>& B) ;
template <typename C> Tensor<C> operator + (Tensor<C>& A, C b) ;

template <typename C> Tensor<C> operator - (Tensor<C>& A, Tensor<C>& B) ;
template <typename C> Tensor<C> operator - (C a, Tensor<C>& B) ;
template <typename C> Tensor<C> operator - (Tensor<C>& A, C b) ;

template <typename C> Tensor<C> operator * (C a, Tensor<C>& B) ;
template <typename C> Tensor<C> operator * (Tensor<C>& A, C b) ;
// with BLAS
Tensor<double> operator * (Tensor<double>& A, Tensor<double>& B) ;

template <typename C> Tensor<C> operator / (Tensor<C>& A, C b) ;
//
template <typename C> bool isAllDimAgree(Tensor<C>& A, Tensor<C>& B) ;
// with BLAS
Tensor<double> contractTensors(Tensor<double>& A, int a1, Tensor<double>& B, int b1) ;
Tensor<double> contractTensors(Tensor<double>& A, int a1, int a2, Tensor<double>& B, int b1, int b2) ;
Tensor<double> contractTensors(Tensor<double>& A, int a1, int a2, int a3, Tensor<double>& B, int b1, int b2, int b3) ;

Tensor<double> contractTensors(Tensor<double> A, Tensor<int> indA_right, Tensor<double> B, Tensor<int> indB_left) ;

Tensor<int> findOuterInd(int rank, Tensor<int> ind_inner) ;

Tensor<int> vertcat(Tensor<int> ind_left, Tensor<int> ind_right) ;
//-------------------------------------------with Lapack------------------------------------------------
// Computes eigenvalues and eigenvectors of a real symmetric matrix using the Relatively Robust Representations.
// A * U = U * D
// info = symEig(A, U, D)
int symEig(Tensor<double> A, Tensor<double>& U, Tensor<double>& D) ;
int symEig(Tensor<double> A, Tensor<double>& D) ;

// Computes the singular value decomposition of a general rectangular matrix using a divide and conquer method.
// A = U * S * VT
// info = svd(A, U, S, VT) ;
int svd(Tensor<double> A, Tensor<double>& U, Tensor<double>& S, Tensor<double>& VT) ;

// QR factorization
// Computes the QR factorization of a general m-by-n matrix.
int qr(Tensor<double>& A, Tensor<double>& R) ;

//======================================================================================================
//======================================================================================================

// constructor
template <typename C>
Tensor<C>::Tensor()
{
	R = -1 ;
	D = NULL ;
	N = 0 ;
	V = NULL ;
}
// overload copy constructor
template <typename C>
Tensor<C>::Tensor(const Tensor<C>& T)
{
	R = -1 ;
	D = NULL ;
	N = 0 ;
	V = NULL ;

	copy(T) ;
}
// create a d0 dimensional column vector, which is represented as a d0 X 1 matrix.
template <typename C>
Tensor<C>::Tensor(int d0)
{
	R = 2 ;
	//
	N = d0 ;
	// column vector
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = 1 ; // column vector
	//
	V = new C[N] ;

	*this = 0 ;
}

template <typename C>
Tensor<C>::Tensor(int d0, int d1)
{
	R = 2 ;
	//
	N = d0 * d1 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	//
	V = new C[N] ;

	*this = 0 ;
}
template <typename C>
Tensor<C>::Tensor(int d0, int d1, int d2)
{
	R = 3 ;
	//
	N = d0 * d1 * d2 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	D[2] = d2 ;
	//
	V = new C[N] ;

	*this = 0 ;
}

template <typename C>
Tensor<C>::Tensor(const std::vector<int>& dim)
{
	if (dim.size() == 1) // column vector
	{
		R = 2 ;
		//
		N = dim[0] ;
		//
		D = new int[R] ;
		D[0] = dim[0] ;
		D[1] = 1 ;
	}
	else
	{
		R = dim.size() ;
		//
		N = 1 ;
		for (int i = 0; i < R; i ++)
		{
			N = N * dim[i] ;
		}
		//
		D = new int[R] ;
		for (int i = 0; i < R; i ++)
		{
			D[i] = dim[i] ;
		}
	}

	//
	V = new C[N] ;

	*this = 0 ;
}

template <typename C>
Tensor<C>::Tensor(const Tensor<int>& dim, C x)
{
	if (dim.numel() == 1) // vector
	{
		R = 2 ;
		//
		N = dim[1] ;
		// column vector
		D = new int[R] ;
		D[0] = N ;
		D[1] = 1 ; // column vector
	}
	else
	{
		R = dim.numel() ;
		//
		N = dim.prod() ;
		//
		D = new int[R] ;
		for (int i = 0; i < R; i ++)
		{
			D[i] = dim[i + 1] ;
		}
	}

	//
	V = new C[N] ;

	*this = x ;
}

//     destructor
template <typename C>
Tensor<C>::~Tensor()
{
//	std::cout << "delete tensor" << std::endl ;
	if (N > 0)
	{
		delete[] V ;
	}
	if (R > 0)
	{
		delete[] D ;
	}
	R = -1 ;
	D = NULL ;
	N = 0 ;
	V = NULL ;
}

template <typename C>
void Tensor<C>::copy(const Tensor<C>& T)
{
//	std::cout << "copy tensor" << std::endl ;
	// if lhs tensor is not empty, delete D
	if (R > 0)
	{
		delete[] D ;
	}
	R = T.rank() ;
	// if rhs tensor is not empty, create D
	if (R > 0)
	{
		D = new int[R] ;
		for (int i = 1; i <= R; i ++)
		{
			D[i - 1] = T.dimension(i) ; // the function "dimension" overload the index starting from 1
		}
	}
	else // if rhs is empty
	{
		D = NULL ;
	}

	// if lhs tensor is not empty, delete V
	if (N > 0)
	{
		delete[] V ;
	}
	N = T.numel() ;
	// if rhs tensor is not empty, create V
	if (N > 0)
	{
		V = new C[ N ] ;
		for (int i = 1; i <= N; i++)
		{
			V[i - 1] = T[i] ;
		}
	}
	else // if rhs is empty
	{
		V = NULL ;
	}
}

//-----------------------------------------------------------------------------------------------------


void Tensor<double>::randUniform()
{
	unsigned int randSeed = time(0) - 13e8 ;

	VSLStreamStatePtr stream;
	vslNewStream( &stream, VSL_BRNG_MT19937, randSeed );
	/* Generating */
	vdRngUniform( VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, N, V, -0.5, 0.5 );
	/* Deleting the stream */
	vslDeleteStream( &stream );
}

template <typename C>
void Tensor<C>::arithmeticSequence(C inital, C difference)
{
	C a = inital ;
	for (int i = 0; i < N; i ++)
	{
		V[i] = a ;
		a = a + difference ;
	}
}

//-----------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// overload operators
// overload "="
template <typename C>
Tensor<C>& Tensor<C>::operator = (const C a)
{
	for (int i = 0; i < N; i++)
	{
		V[i] = a ; // all elements are a
	}
	return *this ;
}
// copy tensor
template <typename C>
Tensor<C>& Tensor<C>::operator = (Tensor<C> T)
{
	copy(T) ;

	return *this ;

}

template <typename C>
C& Tensor<C>::operator [] (int n) const
{
	if (n <= 0)
	{
		std::cout << "operator [] error: index must be a positive integer." << std::endl ;
		exit(0) ;
	}
	else if (n > N)
	{
		std::cout << "operator[] error: Index exceeds matrix dimensions." << std::endl ;
		exit(0) ;
	}
	else
	{
		return V[n - 1] ;
	}
}
//************************************************
// overload "()"
template <typename C>
C& Tensor<C>::operator () (const int i1)
{
	if ((D[0] == 1 || D[1] == 1) && R == 2) // row vector || column vector
	{
		if (N < i1)
		{
			std::cout << "operator(i1) error: Index exceeds matrix dimensions." << std::endl ;
			exit(0) ;
		}
		else
		{
			return (*this)[i1] ;
		}

	}
	else
	{
		std::cout << "operator(i1) error: Number of indices and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
C& Tensor<C>::operator () (const int i1, const int i2)
{
	Tensor<int> position(2) ;
	position[1] = i1 ;
	position[2] = i2 ;

	return at(position) ;
}

template <typename C>
C& Tensor<C>::operator () (const int i1, const int i2, const int i3)
{
	Tensor<int> position(3) ;
	position[1] = i1 ;
	position[2] = i2 ;
	position[3] = i3 ;

	return at(position) ;
}

template <typename C>
C& Tensor<C>::operator () (Tensor<int> ind)
{
	return at(ind) ;
}

// private member function
template <typename C>
C& Tensor<C>::at(Tensor<int> &position)
{

	int p = tensorIndToArrayPos(position) ;
	return V[p - 1] ;
}

template <typename C>
bool Tensor<C>::outOfBound(Tensor<int> &position)
{
	if (position.numel() == 1)
	{
		if (N < position[1])
		{
			return true ;
		}
	}
	else // position.numel() >= 2
	{
		for (int i = 1; i <= R; i++)
		{
			if (D[i-1] < position[i])
			{

				std::cout << "i :" << i << std::endl ;
				std::cout << "position[i] :" << position[i] << std::endl ;
				std::cout << "D[i-1] :" << D[i-1] << std::endl ;
				return true ;
			}
		}
	}

	return false ;
}
//---------------------------------------
// convert array position to tensor index
template <typename C>
Tensor<int> Tensor<C>::arrayPosToTensorInd(int p)
{
	if (p > N)
	{
		std::cout << "arrayPosToTensorInd error: Position exceeds tensor dimensions." << std::endl ;
		exit(0) ;
	}

	// this is a generalized positional notation | 广义进位制转换
	//--------------------------------------------------
	// change start from 1 to start from 0
	p = p - 1 ;
	//--------------------------------------------------
	Tensor<int> ind(R) ; // initialize ind = [0,0,0,...]
	int r = p ;
	for (int i = 0; i < R; i ++)
	{
		int index = r % D[i] ; // 余数
		ind(i + 1) = index ;
		r = r / D[i] ; // 取整的商
	}

	//----------------------------------------
	// change start from 0 to start from 1
	ind = ind + 1 ;
	//----------------------------------------
	return ind ;
}

// convert tensor index to array position
template <typename C>
int Tensor<C>::tensorIndToArrayPos(Tensor<int> ind)
{
	if (ind.numel() == 1)
	{
		if (outOfBound(ind))
		{
			std::cout << "tensorIndToArrayPos error: Index exceeds matrix dimensions." << std::endl ;
			exit(0) ;
		}
		else
		{
			return ind(1) ;
		}
	}
	else // ind.numel() >= 2
	{
		if (R != ind.numel())
		{
			std::cout << "tensorIndToArrayPos error: Number of indices and tensor rank mismatch!" << std::endl ;
			exit(0) ;
		}
		else if (outOfBound(ind))
		{
			std::cout << "tensorIndToArrayPos error: Index exceeds matrix dimensions." << std::endl ;
			exit(0) ;
		}
		else
		{
			//----------------------------------------------
			// change start from 1 to start from 0
			ind = ind - 1 ;
			//----------------------------------------------
			int p = ind(1) ;
			int base = D[0] ;
			for (int i = 1; i < R; i++)
			{
				p = p + base * ind[i + 1] ;
				base = base * D[i] ;
			}
			//----------------------------------------------
			// change start from 0 to start from 1
			p = p + 1 ;
			//----------------------------------------------
			return p ;
		}
	}

}

template <typename C>
Tensor<C> Tensor<C>::subTensor(int start1, int end1, int start2, int end2)
{
	if ( (start1 > end1) || (start2 > end2))
	{
		std::cout << "subTensor error: start index must not larger that end index." << std::endl ;
		exit(0) ;
	}

	Tensor<int> start(2) ;
	Tensor<int> end(2) ;

	start(1) = start1 ;
	start(2) = start2 ;

	end(1) = end1 ;
	end(2) = end2 ;

	return subTensor(start, end) ;
}

template <typename C>
Tensor<C> Tensor<C>::subTensor(Tensor<int>& start, Tensor<int>& end)
{
	if (start.numel() != end.numel())
	{
		std::cout << "length of start and end vectors must be the same!" << std::endl ;
		exit(0) ;
	}
	Tensor<int> dim ;
	//----- create subTensor with all 0 elements: Ts -------
	// create 1D tensor to represent dimensions of Ts
	if (start.numel() >= 2)
	{
		dim = Tensor<int>(start.numel()) ;
		// set dim value
		dim = end - start ;
		dim = dim + 1 ;
	}
	else // T is a vector
	{
		dim = Tensor<int>(2) ;
		if ((*this).dimension(1) == 1) // row vector
		{
			dim(1) = 1 ;
			dim(2) = end(1) - start(1) + 1 ;
		}
		else // column vector
		{
			dim(1) = end(1) - start(1) + 1 ;
			dim(2) = 1 ;
		}
	}
	// create Ts
	Tensor<C> Ts(dim, 0) ;

	if (start.numel() == 1) // T is a vector
	{
		for ( int i = 1 ; i <= Ts.numel(); i ++)
		{
			int indT = i + start(1) - 1 ;
			Ts(i) = (*this)(indT) ;
		}
	}
	else // T is not a vector
	{
		Tensor<int> indTs ;
		Tensor<int> indT ;
		for ( int i = 1; i <= Ts.numel(); i ++)
		{
			indTs = Ts.arrayPosToTensorInd(i) ;
			indT = indTs + start ;
			indT = indT - 1 ;

			Ts(indTs) = (*this)(indT) ;
		}
	}

	return Ts ;

}

// Remove singleton dimensions
template <typename C>
void Tensor<C>::squeeze()
{
	if (R > 2)
	{
		// count number of non-singleton dimensions
		int r = 0 ;
		int* d = new int[R] ;
		for (int i = 0; i < R; i ++)
		{
			if (D[i] > 1)
			{
				d[r] = D[i] ;
				r ++ ;
			}
		}
		//
		delete[] D ;
		if (r > 1)
		{
			D = new int[r] ;
			for (int i = 0; i < r; i ++)
			{
				D[i] = d[i] ;
			}
			R = r ;
		}
		else if (r == 1)
		{
			D = new int[2] ;
			D[0] = d[0] ;
			D[1] = 1 ;

			R = 2 ;
		}
		else // r == 0
		{
			D = new int[2] ;
			D[0] = 1 ;
			D[1] = 1 ;

			R = 2 ;
		}
		delete[] d ;
	}
}
//-----------------------------------------------------------------------

// transpose matrix or vector
template <typename C>
Tensor<C> Tensor<C>::trans()
{
	if ( R == 2)
	{
		int d0 = D[0] ;
		int d1 = D[1] ;

		Tensor<C> T(d1, d0) ; // create a d1 X d0 matrix

		for ( int i = 1; i <= d0; i ++)
		{
			for (int j = 1; j <= d1; j ++)
			{
				T(j, i) = (*this)(i, j) ;
			}
		}

		return T ;
	}
	else
	{
		std::cout << "trans error: transpose on tensor more than 2nd-order is not defined." << std::endl ;
		exit(0) ;
	}

}

// Rearrange dimensions of N-D array

template <typename C>
Tensor<C> Tensor<C>::permute(int i1, int i2)
{
	Tensor<int> order(2) ;
	order(1) = i1 ;
	order(2) = i2 ;

	return permute(order) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(int i1, int i2, int i3)
{
	Tensor<int> order(3) ;
	order(1) = i1 ;
	order(2) = i2 ;
	order(3) = i3 ;

	return permute(order) ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(Tensor<int> order)
{
	if (order.min() <= 0)
	{
		std::cout << "permute error: permutation indices must be positive integers!" << std::endl ;
		exit(0) ;
	}

	if (order.numel() != R)
	{
		std::cout << "permute error: array dimension and the extent of ORDER must be the same! " << std::endl ;
		exit (0) ;
	}

	Tensor<int> dim(R) ;
	for (int i = 1; i <= R; i ++)
	{
		dim(i) = dimension(order(i)) ;
	}
	// create T
	Tensor<C> T(dim, 0) ;

	Tensor<int> indT0(R) ;
	Tensor<int> indT(R) ;
	for (int i = 1; i <= N; i ++)
	{
		indT0 = arrayPosToTensorInd(i) ;
		for (int j = 1; j <= R; j ++)
		{
			indT(j) = indT0(order(j)) ;
		}
		T(indT) = (*this)[i] ;
	}
	return T ;
}

// Reshape tensor

template <typename C>
Tensor<C> Tensor<C>::reshape(int i1)
{
	Tensor<int> dim(1) ;
	dim(1) = i1 ;

	return reshape(dim) ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(int i1, int i2)
{
	Tensor<int> dim(2) ;
	dim(1) = i1 ;
	dim(2) = i2 ;

	return reshape(dim) ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(int i1, int i2, int i3)
{
	Tensor<int> dim(3) ;
	dim(1) = i1 ;
	dim(2) = i2 ;
	dim(3) = i3 ;

	return reshape(dim) ;
}

template <typename C>
Tensor<C> Tensor<C>::reshape(Tensor<int> dim)
{
	if (dim.prod() != N)
	{
		std::cout << "reshape error: the number of elements must not change." << std::endl ;
		exit(0) ;
	}
	Tensor<C> T(dim, 0) ;

	for (int i = 1; i <= N; i ++)
	{
		T[i] = (*this)[i] ;
	}
	return T ;
}

//-----------------------------------------------------------------------
template <typename C>
C Tensor<C>::min()
{
	C x = V[0] ;
	for (int i = 1; i < N; i ++)
	{
		if (x > V[i])
		{
			x = V[i] ;
		}
	}
	return x ;
}

/*
 * this function is only suitable for Tensor<double>
 * normType
 * CHARACTER*1. Specifies the value to be returned by the routine:
 * = 'M' or 'm': val = max(abs(Aij)), largest absolute value of the matrix A.
 * = '1' or 'O' or 'o': val = norm1(A), 1-norm of the matrix A (maximum column sum),
 * = 'I' or 'i': val = normI(A), infinity norm of the matrix A (maximum row sum),
 * = 'F', 'f', 'E' or 'e': val = normF(A), Frobenius norm of the matrix A (square root of sum of squares).
 */
double Tensor<double>::norm(char normType)
{
	if (R > 2)
	{
		std::cout << "norm error: norm of tensor with rank > 2 is undefined." << std::endl ;
		exit(0) ;
	}
	double val ;
	if ((D[0] == 1 || D[1] == 1)) // vector
	{
		int incx = 1 ;
		switch (normType)
		{
		case '1':
			val = dasum(&N, V, &incx) ;
			break ;
		case '2':
			val = dnrm2(&N, V, &incx) ;
			break ;
		}
	}
	else // matrix
	{
		int m = D[0] ;
		int n = D[1] ;
		double* work = new double[m] ;
		val = dlange(&normType, &m, &n, V, &m, work) ;
		delete [] work ;
	}
	return val ;
}
//-----------------------------------------------------------------------
template <typename C>
int Tensor<C>::numel() const
{
	return N ;
}

template <typename C>
int Tensor<C>::rank() const
{
	return R ;
}

template <typename C>
int* Tensor<C>::dimArr() const
{
	return D ;
}

// return the pointer to the array of value
template <typename C>
C* Tensor<C>::ValArr() const
{
	return V ;
}

template <typename C>
int Tensor<C>::dimension(int i) const
{
	return D[i - 1] ;
}

template <typename C>
C Tensor<C>::prod() const
{
	C x = 1 ;
	for (int i = 0; i < N; i ++)
	{
		x = x * V[i] ;
	}
	return x ;
}

//
template <typename C>
void Tensor<C>::display()
{
	if (R > 0)
	{
		if (R == 2)
		{
//			std::cout << "display : " << std::endl ;
			printMatrix() ;
		}
	}
	else
	{
		std::cout << "this is an empty tensor" << std::endl ;
	}
	std::cout << std::endl ;
}

template <typename C>
void Tensor<C>::printMatrix()
{
	for (int i = 1; i <= D[0]; i ++)
	{
//		std::cout << "i = " << i << std::endl ;
		for (int j = 1; j <= D[1]; j ++)
		{
//			std::cout << "j = " << j << std::endl ;
			std::cout << std::setw(14) << (*this)(i, j) << ',' ;
		}
		std::cout << std::endl ;
	}
}
//------------------------------------------------------------------

// non-member functions

// overload operators
template <typename C>
Tensor<C> operator + (Tensor<C>& A, Tensor<C>& B)
{

	if (isAllDimAgree(A, B)) // check if A and B are tensors with same size(rank and dim)
	{
		Tensor<C> T = A ;
		for (int i = 1; i <= A.numel(); i++)
		{
			T[i] = A[i] + B[i] ;
		}
		return T ;
	}
	else
	{
		std::cout << "Tensor operator + error: tensor dimensions must agree." << std::endl ;
		exit (0) ;
	}
}

template <typename C>
Tensor<C> operator + (C a, Tensor<C>& B)
{
	Tensor<C> T = B ;
	for (int i = 1; i <= B.numel(); i++)
	{
		T[i] = a + B[i] ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator + (Tensor<C>& A, C b)
{
	Tensor<C> T = A ;
	for ( int i = 1; i <= A.numel(); i++)
	{
		T[i] = A[i] + b ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator - (Tensor<C>& A, Tensor<C>& B)
{

	if (isAllDimAgree(A, B)) // check if A and B are tensors with same size(rank and dim)
	{
		Tensor<C> T = A ;
		for (int i = 1; i <= A.numel(); i++)
		{
			T[i] = A[i] - B[i] ;
		}
		return T ;
	}
	else
	{
		std::cout << "Tensor operator - error: tensor dimensions must agree." << std::endl ;
		exit (0) ;
	}
}

template <typename C>
Tensor<C> operator - (C a, Tensor<C>& B)
{
	Tensor<C> T = B ;
	for (int i = 1; i <= B.numel(); i++)
	{
		T[i] = a - B[i] ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator - (Tensor<C>& A, C b)
{
	Tensor<C> T = A ;
	for ( int i = 1; i <= A.numel(); i++)
	{
		T[i] = A[i] - b ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator * (C a, Tensor<C>& B)
{
	Tensor<C> T = B ;
	for (int i = 1; i <= B.numel(); i++)
	{
		T[i] = a * B[i] ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator * (Tensor<C>& A, C b)
{
	Tensor<C> T = A ;
	for ( int i = 1; i <= A.numel(); i++)
	{
		T[i] = A[i] * b ;
	}
	return T ;
}

Tensor<double> operator * (Tensor<double>& A, Tensor<double>& B)
{
	if (A.rank() != 2) // A is not a matrix
	{
		std::cout << "operator * error: 1st input must be a matrix." << std::endl ;
		exit(0) ;
	}
	if (B.rank() != 2) // B is not a matrix
	{
		std::cout << "operator * error: 2nd input must be a matrix." << std::endl ;
		exit(0) ;
	}

	int m = A.dimension(1) ;
	int k = A.dimension(2) ;

	int ldB = B.dimension(1) ;
	int n = B.dimension(2) ;

	if (k != ldB)
	{
		std::cout << "operator * error: Inner matrix dimensions must agree." << std::endl ;
		exit(0) ;
	}

	Tensor<double> AB(m, n) ;

	// AB = alpha*A*B + beta*AB
	char transa = 'N' ;
	char transb = 'N' ;
	double alpha = 1.0 ;
	double beta = 0.0 ;
	dgemm(&transa, &transb, &m, &n, &k, &alpha, &(A[1]), &m, &(B[1]), &k, &beta, &(AB[1]), &m) ;

	return AB ;
}

template <typename C>
Tensor<C> operator / (Tensor<C>& A, C b)
{
	Tensor<C> T = A ;
	for ( int i = 1; i <= A.numel(); i++)
	{
		T[i] = A[i] / b ;
	}
	return T ;
}

template <typename C>
bool isAllDimAgree(Tensor<C>& A, Tensor<C>& B)
{
	if (A.rank() != B.rank())
	{
		return false ;
	}
	for (int i = 1; i <= A.rank(); i++)
	{
		if (A.dimension(i) != B.dimension(i))
		{
			return false ;
		}
	}
	return true ;
}


Tensor<double> contractTensors(Tensor<double>& A, int a1, Tensor<double>& B, int b1)
{
	Tensor<int> indA_right(1) ;
	indA_right(1) = a1 ;

	Tensor<int> indB_left(1) ;
	indB_left(1) = b1 ;

	return contractTensors(A, indA_right, B, indB_left) ;

}

Tensor<double> contractTensors(Tensor<double>& A, int a1, int a2, Tensor<double>& B, int b1, int b2)
{
	Tensor<int> indA_right(2) ;
	indA_right(1) = a1 ;
	indA_right(2) = a2 ;

	Tensor<int> indB_left(2) ;
	indB_left(1) = b1 ;
	indB_left(2) = b2 ;

	return contractTensors(A, indA_right, B, indB_left) ;

}

Tensor<double> contractTensors(Tensor<double>& A, int a1, int a2, int a3, Tensor<double>& B, int b1, int b2, int b3)
{
	Tensor<int> indA_right(3) ;
	indA_right(1) = a1 ;
	indA_right(2) = a2 ;
	indA_right(3) = a3 ;

	Tensor<int> indB_left(3) ;
	indB_left(1) = b1 ;
	indB_left(2) = b2 ;
	indB_left(3) = b3 ;

	return contractTensors(A, indA_right, B, indB_left) ;

}

Tensor<double> contractTensors(Tensor<double> A, Tensor<int> indA_right, Tensor<double> B, Tensor<int> indB_left)
{
	// A(indA_left, indA_right)
	// if indA_right.numel() == A.rank(), indA_left is an empty tensor
	Tensor<int> indA_left = findOuterInd(A.rank(), indA_right) ;
	// B(indB_left, indB_right)
	// if indB_left.numel() == B.rank(), indB_right is an empty tensor
	Tensor<int> indB_right = findOuterInd(B.rank(), indB_left) ;

	Tensor<int> indA = vertcat(indA_left, indA_right) ;
	Tensor<int> indB = vertcat(indB_left, indB_right) ;

	A = A.permute(indA) ;
	B = B.permute(indB) ;
	//---------------------
	// create dim of AB
	int rankAB = indA_left.numel() + indB_right.numel() ;
	Tensor<int> dimAB ;
	if (rankAB < 2) // 1, 0
	{
		dimAB = Tensor<int>(2) ;
	}
	else
	{
		dimAB = Tensor<int>(rankAB) ;
	}
	dimAB = 1 ; // initialize all dim to 1

	int id = 1 ; // iteration to find dim of AB
	// compute matrix product
	int m = 1 ; // left dimension of A
	// if indA_left is an empty tensor, m = 1
	for (int i = 1; i <= indA_left.numel(); i ++)
	{
		m = m * A.dimension(i) ;
		dimAB(id) = A.dimension(i) ;
		id ++ ;
	}

	if (id == 1 && rankAB < 2)
	{
		id ++ ;
	}

	int k = 1 ; // right dimension A
	for(int i = (indA_left.numel() + 1); i <= A.rank() ; i ++)
	{
		k = k * A.dimension(i) ;
	}

	int ldB = 1 ; // left dimension of B
	for (int i = 1; i <= indB_left.numel(); i ++)
	{
		ldB = ldB * B.dimension(i) ;
	}

	if (k != ldB)
	{
		std::cout << "contractTensors error: Inner matrix dimensions must agree." << std::endl ;
		exit(0) ;
	}

	int n = 1 ; // right dimension of B
	// if indB_right is an empty tensor, n = 1
	for (int i = (indB_left.numel() + 1); i <= B.rank(); i ++)
	{
		n = n * B.dimension(i) ;
		dimAB(id) = B.dimension(i) ;
		id ++ ;
	}

	Tensor<double> AB(dimAB, 0) ;

	// AB = alpha*A*B + beta*AB
	char transa = 'N' ;
	char transb = 'N' ;
	double alpha = 1.0 ;
	double beta = 0.0 ;
	dgemm(&transa, &transb, &m, &n, &k, &alpha, &(A[1]), &m, &(B[1]), &k, &beta, &(AB[1]), &m) ;

	return AB ;
}

Tensor<int> findOuterInd(int rank, Tensor<int> ind_inner)
{
	bool existence = false ;
	int k = 1 ;
	Tensor<int> ind_outer(rank - ind_inner.numel()) ;

	for (int i = 1; i <= rank; i ++) // iteration for all indices
	{
		for (int j = 1; j <= ind_inner.numel(); j ++ )
		{
			if (i == ind_inner(j)) //
			{
				existence = true ;
				break ;
			}
		}
		if (existence == false)
		{
			ind_outer(k) = i ;
			k ++ ;
		}
		else
		{
			existence = false ;
		}
	}
	return ind_outer ;
}

Tensor<int> vertcat(Tensor<int> ind_left, Tensor<int> ind_right)
{
	int rank = ind_left.numel() + ind_right.numel() ;
	Tensor<int> ind(rank) ;

	int j = 1 ;
	for (int i = 1; i <= ind_left.numel(); i ++)
	{

		ind(j) = ind_left(i) ;
		j ++ ;
	}
	for (int i = 1; i <= ind_right.numel(); i ++)
	{
		ind(j) = ind_right(i) ;
		j ++ ;
	}
	return ind ;
}

//-------------------------------------------with Lapack------------------------------------------------
// Computes eigenvalues and eigenvectors of a real symmetric matrix using the Relatively Robust Representations.
// A * Z = Z * W
int symEig(Tensor<double> A, Tensor<double>& Z, Tensor<double>& W)
{
	if (A.rank() != 2)
	{
		std::cout << "symEig error: 1st input must be a matrix." << std::endl ;
		exit(0) ;
	}
	if (A.dimension(1) != A.dimension(2))
	{
		std::cout << "symEig error: 1st input matrix must be square." << std::endl ;
		exit(0) ;
	}

	char jobz = 'V' ; // eigenvalues and eigenvectors are computed.
	char range = 'A' ; // the routine computes all eigenvalues.
	char uplo = 'U' ; // a stores the upper triangular part of A.

	int n = A.dimension(1) ; // The order of the matrix A (n ≥ 0).

	int lda = n ;

	double vl = 0.0 ;
	double vu = 0.0 ;

	int il = 0 ;
	int iu = 0 ;

	double abstol = 0.0 ; // The absolute error tolerance to which each eigenvalue/eigenvector is required.

	int m = n ;

	W = Tensor<double>(n) ;
	Z = Tensor<double>(n, n) ;

	int ldz = n ;

	Tensor<int> isuppz(2 * n) ;

	//************************************************
	// workspace query
	Tensor<double> work(1) ;
	int lwork = - 1 ;
	Tensor<int> iwork(1) ;
	int liwork = -1 ;

	int info ;

	dsyevr(&jobz, &range, &uplo, &n, &(A[1]), &lda, &vl, &vu, &il, &iu, &abstol, &m,
			&(W[1]), &(Z[1]), &ldz, &(isuppz[1]), &(work[1]), &lwork, &(iwork[1]), &liwork, &info) ;

	//********************************************************************
	// computation
	lwork = (int)work(1) ;
	work = Tensor<double>(lwork) ;
	liwork = iwork(1) ;
	iwork = Tensor<int>(liwork) ;

	dsyevr(&jobz, &range, &uplo, &n, &(A[1]), &lda, &vl, &vu, &il, &iu, &abstol, &m,
			&(W[1]), &(Z[1]), &ldz, &(isuppz[1]), &(work[1]), &lwork, &(iwork[1]), &liwork, &info) ;

	if (info != 0)
	{
		std::cout << "info = " << info << std::endl ;
	}

	return info ;
}

// Computes the singular value decomposition of a general rectangular matrix using a divide and conquer method.
// A = U * S * VT
// info = svd(A, U, S, VT) ;
int svd(Tensor<double> A, Tensor<double>& U, Tensor<double>& S, Tensor<double>& VT)
{
	if (A.rank() != 2)
	{
		std::cout << "svd error: 1st input must be a matrix." << std::endl ;
		exit(0) ;
	}

	char jobz = 'S' ;

	int m = A.dimension(1) ; // The number of rows of the matrix A
	int n = A.dimension(2) ; // The number of columns in A

	int lda = m ; // The leading dimension of  A
	int ldu = m ; // The leading dimensions of U
	int ldvt = std::min(m, n) ; // The leading dimensions of VT

	S = Tensor<double>(ldvt) ;
	U = Tensor<double>(m, ldvt) ;
	VT = Tensor<double>(ldvt, n) ;

	Tensor<int> iwork(8 * ldvt) ;

	int info ;
	//************************************************
	// workspace query
	Tensor<double> work(1) ;
	int lwork = - 1 ;

	dgesdd(&jobz, &m, &n, &(A[1]), &lda, &(S[1]), &(U[1]), &ldu, &(VT[1]), &ldvt, &(work[1]), &lwork, &(iwork[1]), &info) ;

	//*********************************************************************
	// computation
	lwork = (int)work(1) ;
	work = Tensor<double>(lwork) ;

	dgesdd(&jobz, &m, &n, &(A[1]), &lda, &(S[1]), &(U[1]), &ldu, &(VT[1]), &ldvt, &(work[1]), &lwork, &(iwork[1]), &info) ;

	if (info != 0)
	{
		std::cout << "info = " << info << std::endl ;
	}

	return info ;
}

// Computes the QR factorization of a general m-by-n matrix.
// A = Q * R
int qr(Tensor<double>& A, Tensor<double>& R)
{
	if (A.rank() != 2)
	{
		std::cout << "qr error: 1st input must be a matrix." << std::endl ;
		exit(0) ;
	}

	int m = A.dimension(1) ; // The number of rows in the matrix A
	int n = A.dimension(2) ; // The number of columns in A

	int k = std::min(m, n) ;

	R = A ;

	int lda = m ;

	Tensor<double> tau(k) ;

	int info ;
	//************************************************
	// workspace query
	Tensor<double> work(1) ;
	int lwork = - 1 ;

	dgeqrf(&m, &n, &(R[1]), &lda, &(tau[1]), &(work[1]), &lwork, &info) ;

	//************************************************
	// computation
	lwork = (int)work(1) ;
	work = Tensor<double>(lwork) ;

	dgeqrf(&m, &n, &(R[1]), &lda, &(tau[1]), &(work[1]), &lwork, &info) ;

	if (info != 0)
	{
		std::cout << "info = " << info << std::endl ;
	}

	if ( m > n)
	{
		R = R.subTensor(1,n, 1,n) ;
	}

	for (int i = 1; i <= R.dimension(1); i ++)
	{
		for (int j = 1; i > j ; j ++)
		{
			R(i, j) = 0 ;
		}
	}

	return info ;
}
//***********************************************************************

//---------------------------------------
#endif /* TENSOR_HPP_ */
