/*
 * Tensor.hpp
 *
 *  Created on: 2013-7-16
 *  Updated on: 2014-9-24
 *  Updated on: 2014-10-2
 *  Updated on: 2014-10-15
 *  Updated on: 2014-10-16
 *  Updated on: 2014-10-29
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
#include <omp.h>

#include "Index.h"
#include "auxiliary.hpp"
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
	// tensor indices
	std::vector<Index> _inds ;
	//--------------------------------------------------------------------------------
	void printMatrix() ;
	void printMatrix(int precision) ;
	C& at(Tensor<int> &position) ;
	bool outOfBound(Tensor<int> &position) ;
	bool outOfBound(int * ind) ;
	void copy(const Tensor<C>& T) ;
	void copyValue(C* V1, C* V, int num) ;

public:
	// constructor
	Tensor() ; // empty tensor
	Tensor(const Tensor<C>& T) ; // overload copy constructor
	// create a d0 dimensional column vector, which is represented as a d0 X 1 matrix.
	Tensor(int d0) ; // all elements are 0
	Tensor(int d0, int d1) ;
	Tensor(int d0, int d1, int d2) ;
	Tensor(int d0, int d1, int d2, int d3) ;
	Tensor(int d0, int d1, int d2, int d3, int d4) ;
	Tensor(int d0, int d1, int d2, int d3, int d4, int d5) ;
	Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6) ;
	Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7) ;
	Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8) ;
	Tensor(const std::vector<int>& dim) ;
	Tensor(const Tensor<int>& dim, C x) ;

	// constructor with the input of Index
	Tensor(const Index& x0) ;
	Tensor(const Index& x0, const Index& x1) ;
	Tensor(const Index& x0, const Index& x1, const Index& x2) ;
	Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3) ;
	Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4) ;
	Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4, const Index& x5) ;
	Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4, const Index& x5, const Index& x6) ;
	Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4, const Index& x5, const Index& x6, const Index& x7) ;

	//     destructor
	~Tensor() ;
	//--------------
	void randUniform() ;
	void randUniform(double min, double max) ;
	void arithmeticSequence(C initial, C difference) ;
	void geometricSequence(C initial, C ratio) ;
	//---------------------------------------------------
	// overload operators
	Tensor<C>& operator = (const C a) ;
	Tensor<C>& operator = (const Tensor<C> &T) ; // copy tensor
	C& operator [] (const int n) const ; // get an element by array index without bound check
	C& at(const int n) const ; // get an element by array index with bound check
	// Access operator of an element in Fortran order. Get an element by tensor index
	C& operator () (const int i0) ;
	C& operator () (const int i0, const int i1) ;
	C& operator () (const int i0, const int i1, const int i2) ;
	C& operator () (const int i0, const int i1, const int i2, const int i3) ;
	C& operator () (const int i0, const int i1, const int i2, const int i3, const int i4) ;
	C& operator () (const int i0, const int i1, const int i2, const int i3, const int i4, const int i5) ;
	C& operator () (const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6) ;
	C& operator () (const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7) ;
	C& operator () (const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const int i8) ;
	C& operator () (Tensor<int> ind) ;
	C& operator () (int * ind) ;

	//---------------------------------------------------
	Tensor<C> trans() ; // transpose matrix or vector
	Tensor<C> ctrans() ;

	Tensor<C> permute(int i0, int i1) ;
	Tensor<C> permute(int i0, int i1, int i2) ;
	Tensor<C> permute(int i0, int i1, int i2, int i3) ;
	Tensor<C> permute(int i0, int i1, int i2, int i3, int i4) ;
	Tensor<C> permute(int i0, int i1, int i2, int i3, int i4, int i5) ;
	Tensor<C> permute(int i0, int i1, int i2, int i3, int i4, int i5, int i6) ;
	Tensor<C> permute(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) ;
	Tensor<C> permute(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) ;
	Tensor<C> permute(Tensor<int> &P) ;
	Tensor<C> permute(const int * P) ;

	Tensor<C> permute(const Index& x0, const Index& x1) ;
	Tensor<C> permute(const Index& x0, const Index& x1, const Index& x2) ;
	Tensor<C> permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3) ;
	Tensor<C> permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4) ;
	Tensor<C> permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4, const Index& x5) ;
	Tensor<C> permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4, const Index& x5, const Index& x6) ;
	Tensor<C> permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4, const Index& x5, const Index& x6, const Index& x7) ;
	Tensor<C> permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4,
			              const Index& x5, const Index& x6, const Index& x7, const Index& x8) ;

	void findIndOrd(const std::vector<Index>& inds_new, int * const P) ;

	Tensor<C> reshape(int i0) ;
	Tensor<C> reshape(int i0, int i1) ;
	Tensor<C> reshape(int i0, int i1, int i2) ;
	Tensor<C> reshape(int i0, int i1, int i2, int i3) ;
	Tensor<C> reshape(int i0, int i1, int i2, int i3, int i4) ;
	Tensor<C> reshape(Tensor<int> dim) ;

	Tensor<C> reshape(Index x0) ;
	Tensor<C> reshape(Index x0, Index x1) ;
	Tensor<C> reshape(Index x0, Index x1, Index x2) ;
	Tensor<C> reshape(Index x0, Index x1, Index x2, Index x3) ;
	Tensor<C> reshape(Index x0, Index x1, Index x2, Index x3, Index x4) ;
	Tensor<C> reshape(std::vector<Index> X) ;
	//---------------------------------------------------
	void vecInd_to_NDind(int p, int * ind) ; // convert array position to tensor index
	void NDind_to_vecInd(int * ind, int &p) ; // convert tensor index to array position

	Tensor<int> arrayPosToTensorInd(int p) ; // convert array position to tensor index
	int tensorIndToArrayPos(Tensor<int> ind) ; // convert tensor index to array position

	Tensor<C> subTensor(int start, int end) ;
	Tensor<C> subTensor(int start0, int end0, int start1, int end1) ;
	Tensor<C> subTensor(Tensor<int>& start, Tensor<int>& end) ;

	//Tensor<C> diag(Tensor<C> T) ;
	void squeeze() ;
	//---------------------------------------------------
	Tensor<double> exp() ;
	Tensor<double> sqrt() ;
	Tensor<double> power(double exponent) ;
	C mean() ;
	C min() ;
	C max() ;
	C sum() ;
	C prod() const ;

	double norm(char normType) ;
	double maxAbs() ; // return the absolute value of element with maximum absolute value.

	// Sorts numbers in increasing or decreasing order.
	Tensor<double> sort(char id) ;

	void symmetrize() ;
	bool isSym() ;

	void setSmalltoZero(double a = 1.0e-14) ;
	//---------------------------------------------------
	int rank() const ;
	int numel() const ;
	int size() const { return N ; }
	int* dimArr() const ;
	C* ValArr() const ;
	int dimension(int i) const ;
	//--------------- dealing with index-----------------
	std::vector<Index> inds() const { return _inds ; }
	// define Index to tensor, and assign dim to Index
	void putIndex(std::vector<Index> &a) ;
	void putIndex(Index &x0) ;
	void putIndex(Index &x0, Index &x1) ;
	void putIndex(Index &x0, Index &x1, Index &x2) ;
	void putIndex(Index &x0, Index &x1, Index &x2, Index &x3) ;
	void putIndex(Index &x0, Index &x1, Index &x2, Index &x3, Index &x4) ;
	void putIndex(Index &x0, Index &x1, Index &x2, Index &x3, Index &x4, Index &x5) ;
	void putIndex(Index &x0, Index &x1, Index &x2, Index &x3, Index &x4, Index &x5, Index &x6) ;
	void putIndex(Index &x0, Index &x1, Index &x2, Index &x3, Index &x4, Index &x5, Index &x6, Index &x7) ;

	//--------------------------------------------------
	void display() ;
	void display(int precision) ;
	void display1D() ;
	void info() ; // output tensor information
	//---------------------------------------------------
	// save/load tensor in binary mode
	// 'fileName' include path
	void save(std::string fileName) ;
	void load(std::string fileName) ;


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
inline Tensor<double> operator * (Tensor<double>& A, Tensor<double>& B) ;

template <typename C> Tensor<C> operator / (Tensor<C>& A, C b) ;
//
template <typename C> bool isAllDimAgree(Tensor<C>& A, Tensor<C>& B) ;

inline double compareTensors(Tensor<double>& A, Tensor<double>& A0) ;
//------------------------- with BLAS --------------------------------
template <typename C> C dot(Tensor<C> &A, Tensor<C> &B) ;

inline Tensor<double> contractTensors(Tensor<double>& A, Tensor<double>& B) ;
inline std::vector<Index> findCommonIndex(std::vector<Index> indsA, Tensor<int> &indA_ord_right,
		                        	            std::vector<Index> indsB, Tensor<int> &indB_ord_left) ;

inline Tensor<double> contractTensors(Tensor<double>& A, int a0, Tensor<double>& B, int b0) ;
inline Tensor<double> contractTensors(Tensor<double>& A, int a0, int a1, Tensor<double>& B, int b0, int b1) ;
inline Tensor<double> contractTensors(Tensor<double>& A, int a0, int a1, int a2, Tensor<double>& B, int b0, int b1, int b2) ;
inline Tensor<double> contractTensors(Tensor<double>& A, int a0, int a1, int a2, int a3,
		                                  Tensor<double>& B, int b0, int b1, int b2, int b3) ;

inline Tensor<double> contractTensors(Tensor<double> &A, Tensor<int> indA_right, Tensor<double> &B, Tensor<int> indB_left) ;

inline Tensor<int> findOuterInd(int rank, Tensor<int> ind_inner) ;

inline Tensor<int> vertcat(Tensor<int> ind_left, Tensor<int> ind_right) ;

inline Tensor<double> absorbVector(Tensor<double>& A, int n, Tensor<double>& V) ;

inline Tensor<double> spitVector(Tensor<double>& A, int n, Tensor<double>& V) ;

inline Tensor<double> tensor2xVec(Tensor<double>& A, int n, Tensor<double>& V) ;
inline Tensor<double> tensor3xVec(Tensor<double>& A, int n, Tensor<double>& V) ;
inline Tensor<double> tensor4xVec(Tensor<double>& A, int n, Tensor<double>& V) ;
inline Tensor<double> tensor5xVec(Tensor<double>& A, int n, Tensor<double>& V) ;

//-------------------------------------------with Lapack------------------------------------------------
// Computes eigenvalues and eigenvectors of a real symmetric matrix using the Relatively Robust Representations.
// A * U = U * D
// info = symEig(A, U, D)
inline int symEig(Tensor<double> A, Tensor<double>& U, Tensor<double>& D) ;
inline int symEig(Tensor<double> A, Tensor<double>& D) ;

// input:
// A (real non-symmetric matrix)
// output:
// Wr: column vector which contains real part of eigenvalues
// Wi: column vector which contains imaginary part of eigenvalues
// Vl: left eigenvectors are stored in the columns, complex conjugate pair are Vl(:,j)+i*Vl(:,j+1) and Vl(:,j)-i*Vl(:,j+1)
// Vr: right eigenvectors are stored in the columns, complex conjugate pair are Vr(:,j)+i*Vr(:,j+1) and Vr(:,j)-i*Vr(:,j+1)
inline int reNonSymEig(Tensor<double> A, Tensor<double> &Wr, Tensor<double> &Wi, Tensor<double> &Vl, Tensor<double> &Vr) ;

// Computes the singular value decomposition of a general rectangular matrix using a divide and conquer method.
// A = U * S * V'
// info = svd(A, U, S, V) ;
inline int svd(Tensor<double> A, Tensor<double>& U, Tensor<double>& S, Tensor<double>& V) ;


// QR factorization
// Computes the QR factorization of a general m-by-n matrix.
inline int qr(Tensor<double>& A, Tensor<double>& R) ;
// LQ factorization
// Computes the LQ factorization of a general m-by-n matrix.
inline int lq(Tensor<double>& A, Tensor<double>& L) ;

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

//	*this = 0 ;
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

//	*this = 0 ;
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

//	*this = 0 ;
}

template <typename C>
Tensor<C>::Tensor(int d0, int d1, int d2, int d3)
{
	R = 4 ;
	//
	N = d0 * d1 * d2 * d3 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	D[2] = d2 ;
	D[3] = d3 ;
	//
	V = new C[N] ;

//	*this = 0 ;
}

template <typename C>
Tensor<C>::Tensor(int d0, int d1, int d2, int d3, int d4)
{
	R = 5 ;
	//
	N = d0 * d1 * d2 * d3 * d4 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	D[2] = d2 ;
	D[3] = d3 ;
	D[4] = d4 ;
	//
	V = new C[N] ;

//	*this = 0 ;
}

template <typename C>
Tensor<C>::Tensor(int d0, int d1, int d2, int d3, int d4, int d5)
{
	R = 6 ;
	//
	N = d0 * d1 * d2 * d3 * d4 * d5 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	D[2] = d2 ;
	D[3] = d3 ;
	D[4] = d4 ;
	D[5] = d5 ;
	//
	V = new C[N] ;

//	*this = 0 ;
}

template <typename C>
Tensor<C>::Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6)
{
	R = 7 ;
	//
	N = d0 * d1 * d2 * d3 * d4 * d5 * d6 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	D[2] = d2 ;
	D[3] = d3 ;
	D[4] = d4 ;
	D[5] = d5 ;
	D[6] = d6 ;
	//
	V = new C[N] ;
}

template <typename C>
Tensor<C>::Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7)
{
	R = 8 ;
	//
	N = d0 * d1 * d2 * d3 * d4 * d5 * d6 * d7 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	D[2] = d2 ;
	D[3] = d3 ;
	D[4] = d4 ;
	D[5] = d5 ;
	D[6] = d6 ;
	D[7] = d7 ;
	//
	V = new C[N] ;
}

template <typename C>
Tensor<C>::Tensor(int d0, int d1, int d2, int d3, int d4, int d5, int d6, int d7, int d8)
{
	R = 9 ;
	//
	N = d0 * d1 * d2 * d3 * d4 * d5 * d6 * d7 * d8 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	D[2] = d2 ;
	D[3] = d3 ;
	D[4] = d4 ;
	D[5] = d5 ;
	D[6] = d6 ;
	D[7] = d7 ;
	D[8] = d8 ;
	//
	V = new C[N] ;
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

//	*this = 0 ;
}

template <typename C>
Tensor<C>::Tensor(const Tensor<int>& dim, C x)
{
	if (dim.numel() == 1) // vector
	{
		R = 2 ;
		//
		N = dim[0] ;
		// column vector
		D = new int[R] ;
		D[0] = N ;
		D[1] = 1 ; // column vector
	}
	else
	{
		R = dim.numel() ;
//		std::cout << "R: " << R << std::endl ;
		//
		N = dim.prod() ;
		//
		D = new int[R] ;
		for (int i = 0; i < R; i ++)
		{
			D[i] = dim[i] ;
		}
	}

	//
	V = new C[N] ;

	*this = x ;
}

template <typename C>
Tensor<C>::Tensor(const Index& x0)
{
	R = 2 ;
	//
	N = x0.dim() ;
	// column vector
	D = new int[R] ;
	D[0] = x0.dim() ;
	D[1] = 1 ; // column vector
	//
	V = new C[N] ;

	_inds.resize(1) ;
	_inds[0] = x0 ;
}
template <typename C>
Tensor<C>::Tensor(const Index& x0, const Index& x1)
{
	R = 2 ;
	//
	N = x0.dim() * x1.dim() ;
	//
	D = new int[R] ;
	D[0] = x0.dim() ;
	D[1] = x1.dim() ;
	//
	V = new C[N] ;

	_inds.resize(2) ;

	_inds[0] = x0 ;
	_inds[1] = x1 ;
}
template <typename C>
Tensor<C>::Tensor(const Index& x0, const Index& x1, const Index& x2)
{
	R = 3 ;
	//
	N = x0.dim() * x1.dim() * x2.dim() ;
	//
	D = new int[R] ;
	D[0] = x0.dim() ;
	D[1] = x1.dim() ;
	D[2] = x2.dim() ;
	//
	V = new C[N] ;

	_inds.resize(3) ;

	_inds[0] = x0 ;
	_inds[1] = x1 ;
	_inds[2] = x2 ;
}
template <typename C>
Tensor<C>::Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3)
{
	R = 4 ;
	//
	N = x0.dim() * x1.dim() * x2.dim() * x3.dim() ;
	//
	D = new int[R] ;
	D[0] = x0.dim() ;
	D[1] = x1.dim() ;
	D[2] = x2.dim() ;
	D[3] = x3.dim() ;
	//
	V = new C[N] ;

	_inds.resize(4) ;
	_inds[0] = x0 ;
	_inds[1] = x1 ;
	_inds[2] = x2 ;
	_inds[3] = x3 ;
}
template <typename C>
Tensor<C>::Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4)
{
	R = 5 ;
	//
	N = x0.dim() * x1.dim() * x2.dim() * x3.dim() * x4.dim() ;
	//
	D = new int[R] ;
	D[0] = x0.dim() ;
	D[1] = x1.dim() ;
	D[2] = x2.dim() ;
	D[3] = x3.dim() ;
	D[4] = x4.dim() ;
	//
	V = new C[N] ;

	_inds.resize(5) ;
	_inds[0] = x0 ;
	_inds[1] = x1 ;
	_inds[2] = x2 ;
	_inds[3] = x3 ;
	_inds[4] = x4 ;
}
template <typename C>
Tensor<C>::Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4,
									const Index& x5)
{
	R = 6 ;
	//
	N = x0.dim() * x1.dim() * x2.dim() * x3.dim() * x4.dim() * x5.dim() ;
	//
	D = new int[R] ;
	D[0] = x0.dim() ;
	D[1] = x1.dim() ;
	D[2] = x2.dim() ;
	D[3] = x3.dim() ;
	D[4] = x4.dim() ;
	D[5] = x5.dim() ;
	//
	V = new C[N] ;

	_inds.resize(6) ;
	_inds[0] = x0 ;
	_inds[1] = x1 ;
	_inds[2] = x2 ;
	_inds[3] = x3 ;
	_inds[4] = x4 ;
	_inds[5] = x5 ;
}
template <typename C>
Tensor<C>::Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4,
									const Index& x5, const Index& x6)
{
	R = 7 ;
	//
	N = x0.dim() * x1.dim() * x2.dim() * x3.dim() * x4.dim() * x5.dim() * x6.dim() ;
	//
	D = new int[R] ;
	D[0] = x0.dim() ;
	D[1] = x1.dim() ;
	D[2] = x2.dim() ;
	D[3] = x3.dim() ;
	D[4] = x4.dim() ;
	D[5] = x5.dim() ;
	D[6] = x6.dim() ;
	//
	V = new C[N] ;

	_inds.resize(7) ;
	_inds[0] = x0 ;
	_inds[1] = x1 ;
	_inds[2] = x2 ;
	_inds[3] = x3 ;
	_inds[4] = x4 ;
	_inds[5] = x5 ;
	_inds[6] = x6 ;
}
template <typename C>
Tensor<C>::Tensor(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4,
								  const Index& x5, const Index& x6, const Index& x7)
{
	R = 8 ;
	//
	N = x0.dim() * x1.dim() * x2.dim() * x3.dim() * x4.dim() * x5.dim() * x6.dim() * x7.dim() ;
	//
	D = new int[R] ;
	D[0] = x0.dim() ;
	D[1] = x1.dim() ;
	D[2] = x2.dim() ;
	D[3] = x3.dim() ;
	D[4] = x4.dim() ;
	D[5] = x5.dim() ;
	D[6] = x6.dim() ;
	D[7] = x7.dim() ;
	//
	V = new C[N] ;

	_inds.resize(8) ;

	_inds[0] = x0 ;
	_inds[1] = x1 ;
	_inds[2] = x2 ;
	_inds[3] = x3 ;
	_inds[4] = x4 ;
	_inds[5] = x5 ;
	_inds[6] = x6 ;
	_inds[7] = x7 ;
}

//     destructor
template <typename C>
Tensor<C>::~Tensor()
{
	//std::cout << "call destructor of Tensor" << std::endl ;
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
//		std::cout << "copy tensor" << std::endl ;
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
		for (int i = 0; i < R; i ++)
		{
			D[i] = T.dimension(i) ;
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

		copyValue(V, T.ValArr(), N) ;
	}
	else // if rhs is empty
	{
		V = NULL ;
	}

	// copy index infomation
	_inds = T.inds() ;

}

template <typename C>
void Tensor<C>::copyValue(C* V1, C* V0, int num)
{
	int i ;
//	#pragma omp parallel for
	for (i = 0; i < num; i ++)
	{
		V1[i] = V0[i] ;
	}
}
//-----------------------------------------------------------------------------------------------------


void Tensor<double>::randUniform()
{
	//	unsigned int randSeed = time(0) - 13e8 ;

	unsigned MKL_INT64 randSeed ;

	mkl_get_cpu_clocks(&randSeed) ;

	VSLStreamStatePtr stream;
	vslNewStream( &stream, VSL_BRNG_MT19937, randSeed );
	/* Generating */
	vdRngUniform( VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, N, V, -0.5, 0.5 );
	/* Deleting the stream */
	vslDeleteStream( &stream );
}

void Tensor<double>::randUniform(double min, double max)
{
	//	unsigned int randSeed = time(0) - 13e8 ;

	unsigned MKL_INT64 randSeed ;

	mkl_get_cpu_clocks(&randSeed) ;

	VSLStreamStatePtr stream;
	vslNewStream( &stream, VSL_BRNG_MT19937, randSeed );
	/* Generating */
	vdRngUniform( VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, N, V, min, max );
	/* Deleting the stream */
	vslDeleteStream( &stream );
}

template <typename C>
void Tensor<C>::arithmeticSequence(C initial, C difference)
{
	C a = initial ;
	for (int i = 0; i < N; i ++)
	{
		V[i] = a ;
		a = a + difference ;
	}
}

template <typename C>
void Tensor<C>::geometricSequence(C initial, C ratio)
{
	C a = initial ;
	C r = ratio ;
	for (int i = 0; i < N; i ++)
	{
		V[i] = a ;
		a = a * r ;
	}
}
//-----------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
// overload operators
// overload "="
template <typename C>
Tensor<C>& Tensor<C>::operator = (const C a)
{
	int i ;
//	#pragma omp parallel for
	for (i = 0; i < N; i++)
	{
		V[i] = a ; // all elements are a
	}
	return *this ;
}
// copy tensor
template <typename C>
Tensor<C>& Tensor<C>::operator = (const Tensor<C> &T)
{
	copy(T) ;

	return *this ;

}

// no bound check
template <typename C>
C& Tensor<C>::operator [] (const int n) const
{
	return V[n] ;
}

// with bound check
template <typename C>
C& Tensor<C>::at(const int n) const
{
	if (n < 0)
	{
		std::cout << "operator [] error: index must be a non-negative integer." << std::endl ;
		exit(0) ;
	}
	else if (n >= N)
	{
		std::cout << "operator[] error: Index exceeds matrix dimensions." << std::endl ;
		exit(0) ;
	}
	else
	{
		return V[n] ;
	}
}

//************************************************
// overload "()"
template <typename C>
C& Tensor<C>::operator () (const int i0)
{
	if ((D[0] == 1 || D[1] == 1) && R == 2) // row vector || column vector
	{
		if (i0 >= N)
		{
			std::cout << "operator(i0) error: Index exceeds matrix dimensions." << std::endl ;
			exit(0) ;
		}
		else
		{
			return V[i0] ;
		}

	}
	else
	{
		std::cout << "operator(i0) error: Number of indices and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
C& Tensor<C>::operator () (const int i0, const int i1)
{
	//++++++++++++++++++++++++++++
	return V[ i0 + i1 * D[0] ] ;
}

template <typename C>
C& Tensor<C>::operator () (const int i0, const int i1, const int i2)
{
	//++++++++++++++++++++++++++
	return V[i0 + i1 * D[0] + i2 * D[0] * D[1] ] ;
}

template <typename C>
C& Tensor<C>::operator () (const int i0, const int i1, const int i2, const int i3)
{
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++
	return V[ i0 + i1 * D[0] + i2 * D[0] * D[1] + i3 * D[0] * D[1] * D[2] ] ;
}

template <typename C>
C& Tensor<C>::operator () (const int i0, const int i1, const int i2, const int i3, const int i4)
{
	int n = i0 ;

	int base = D[0] ;
	n += i1 * base ;

	base = base * D[1] ;
	n += i2 * base ;

	base = base * D[2] ;
	n += i3 * base ;

	base = base * D[3] ;
	n += i4 * base ;

	return V[n] ;
}

template <typename C>
C& Tensor<C>::operator () (const int i0, const int i1, const int i2, const int i3, const int i4, const int i5)
{
	int n = i0 ;

	int base = D[0] ;
	n += i1 * base ;

	base = base * D[1] ;
	n += i2 * base ;

	base = base * D[2] ;
	n += i3 * base ;

	base = base * D[3] ;
	n += i4 * base ;

	base = base * D[4] ;
	n += i5 * base ;

	return V[n] ;
}

template <typename C>
C& Tensor<C>::operator () (const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6)
{
	int n = i0 ;

	int base = D[0] ;
	n += i1 * base ;

	base = base * D[1] ;
	n += i2 * base ;

	base = base * D[2] ;
	n += i3 * base ;

	base = base * D[3] ;
	n += i4 * base ;

	base = base * D[4] ;
	n += i5 * base ;

	base = base * D[5] ;
	n += i6 * base ;

	return V[n] ;
}

template <typename C>
C& Tensor<C>::operator () (const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7)
{
	int n = i0 ;

	int base = D[0] ;
	n += i1 * base ;

	base = base * D[1] ;
	n += i2 * base ;

	base = base * D[2] ;
	n += i3 * base ;

	base = base * D[3] ;
	n += i4 * base ;

	base = base * D[4] ;
	n += i5 * base ;

	base = base * D[5] ;
	n += i6 * base ;

	base = base * D[6] ;
	n += i7 * base ;

	return V[n] ;
}

template <typename C>
C& Tensor<C>::operator () (const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7, const int i8)
{
	int n = i0 ;

	int base = D[0] ;
	n += i1 * base ;

	base = base * D[1] ;
	n += i2 * base ;

	base = base * D[2] ;
	n += i3 * base ;

	base = base * D[3] ;
	n += i4 * base ;

	base = base * D[4] ;
	n += i5 * base ;

	base = base * D[5] ;
	n += i6 * base ;

	base = base * D[6] ;
	n += i7 * base ;

	base = base * D[7] ;
	n += i8 * base ;

	return V[n] ;
}

template <typename C>
C& Tensor<C>::operator () (Tensor<int> ind)
{
	return at(ind) ;
}

template <typename C>
C& Tensor<C>::operator () (int * ind)
{
	int p ;
	NDind_to_vecInd(ind, p) ;
	return V[p] ;
}

// private member function
template <typename C>
C& Tensor<C>::at(Tensor<int> &position)
{

	int p = tensorIndToArrayPos(position) ;
	return V[p] ;
}

template <typename C>
bool Tensor<C>::outOfBound(Tensor<int> &position)
{
	if (position.numel() == 1)
	{
		if ( N <= position.at(0) )
		{
			return true ;
		}
	}
	else // position.numel() >= 2
	{
		for (int i = 0; i < R; i++)
		{
			if (D[i] <= position.at(i))
			{

				std::cout << "i :" << i << std::endl ;
				std::cout << "position[i] :" << position[i] << std::endl ;
				std::cout << "D[i] :" << D[i] << std::endl ;
				return true ;
			}
		}
	}

	return false ;
}

template <typename C>
bool Tensor<C>::outOfBound(int * ind)
{
	if (R == 2 && ( D[0] == 1 || D[1] == 1 )) // vector
	{
		if ( N <= ind[0] )
		{
			return true ;
		}
	}
	else //
	{
		for (int i = 0; i < R; i++)
		{
			if (D[i] <= ind[i])
			{
				std::cout << "i :" << i << std::endl ;
				std::cout << "position[i] :" << ind[i] << std::endl ;
				std::cout << "D[i] :" << D[i] << std::endl ;
				return true ;
			}
		}
	}

	return false ;
}
//---------------------------------------
// convert vector index to N-D indices
template <typename C>
void Tensor<C>::vecInd_to_NDind(int p, int * ind)
{
	// this is a generalized positional notation |
	//--------------------------------------------------
	int i ;
	for (i = 0; i < (R-1); i ++)
	{
		ind[i] = p % D[i] ; //
		p /= D[i] ; //
	}
	ind[R-1] = p ;
	//----------------------------------------
}

// convert R-D indices to vector index (R > 1)
template <typename C>
void Tensor<C>::NDind_to_vecInd(int * ind, int &p)
{
	//----------------------------------------------
	p = ind[0] ;
	int base = D[0] ;
	for (int i = 1; i < (R - 1); i++)
	{
		p += base * ind[i] ;
		base *= D[i] ;
	}
	p += base * ind[R-1] ;
	//----------------------------------------------
//	p = ind[0] ;
//	int base = D[0] ;
//	for (int i = 1; i < R ; i++)
//	{
//		p += base * ind[i] ;
//		base *= D[i] ;
//	}
}

//---------------------------------------
// convert array position to tensor index
template <typename C>
Tensor<int> Tensor<C>::arrayPosToTensorInd(int p)
{
	// this is a generalized positional notation |
	//--------------------------------------------------
	Tensor<int> ind(R) ;
	//--------------------------------------------------
	int i ;
	for (i = 0; i < R; i ++)
	{
		ind[i] = p % D[i] ; //
		p /= D[i] ; //
	}
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
			return ind[0] ;
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
			int p = ind[0] ;
			int base = D[0] ;
			for (int i = 1; i < R; i++)
			{
				p += base * ind[i] ;
				base *= D[i] ;
			}
			//----------------------------------------------
			return p ;
		}
	}
}

template <typename C>
Tensor<C> Tensor<C>::subTensor(int start0, int end0)
{
	if ( start0 > end0 )
	{
		std::cout << "subTensor error: start index must not larger that end index." << std::endl ;
		exit(0) ;
	}

	Tensor<int> start(1) ;
	Tensor<int> end(1) ;

	start(0) = start0 ;

	end(0) = end0 ;

	return subTensor(start, end) ;
}

template <typename C>
Tensor<C> Tensor<C>::subTensor(int start0, int end0, int start1, int end1)
{
	if ( (start0 > end0) || (start1 > end1))
	{
		std::cout << "subTensor error: start index must not larger that end index." << std::endl ;
		exit(0) ;
	}

	Tensor<int> start(2) ;
	Tensor<int> end(2) ;

	start(0) = start0 ;
	start(1) = start1 ;

	end(0) = end0 ;
	end(1) = end1 ;

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
		if ((*this).dimension(0) == 1) // row vector
		{
			dim(0) = 1 ;
			dim(1) = end(0) - start(0) + 1 ;
		}
		else // column vector
		{
			dim(0) = end(0) - start(0) + 1 ;
			dim(1) = 1 ;
		}
	}
	//***********
	// create Ts
	Tensor<C> Ts(dim, 0) ;
	if (start.numel() == 1) // T is a vector
	{
		for ( int i = 0 ; i < Ts.numel(); i ++)
		{
			int indT = i + start(0) ;
			Ts(i) = (*this)(indT) ;
		}
	}
	else // T is not a vector
	{
		Tensor<int> indTs ;
		Tensor<int> indT ;
		for ( int i = 0; i < Ts.numel(); i ++)
		{
			indTs = Ts.arrayPosToTensorInd(i) ;
			indT = indTs + start ;
			Ts(indTs) = (*this)(indT) ;
		}
	}
	// copy index
	Ts.putIndex(_inds) ;

	return Ts ;
}

// Remove singleton dimensions
template <typename C>
void Tensor<C>::squeeze()
{

	if (R > 2)
	{
		std::vector<Index> ind_new ;

		// count number of non-singleton dimensions
		int r = 0 ;
		int* d = new int[R] ;
		for (int i = 0; i < R; i ++)
		{
			if (D[i] > 1)
			{
				d[r] = D[i] ;
				r ++ ;

				if ( _inds.size() == R ) // space has been allocated for "_inds"
				{
					ind_new.push_back(_inds[i]) ;
				}
			}
		}
		_inds = ind_new ;
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
	if ( R == 2 )
	{
		return permute(1, 0) ;
	}
	else
	{
		std::cout << "trans error: transpose on tensor more than 2nd-order is not defined." << std::endl ;
		exit(0) ;
	}
}

// Rearrange dimensions of N-D array
template <typename C>
Tensor<C> Tensor<C>::permute(int i0, int i1)
{
	int P[2] ;
	P[0] = i0 ;
	P[1] = i1 ;

	return permute(P) ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(int i0, int i1, int i2)
{
	int P[3] ;
	P[0] = i0 ;
	P[1] = i1 ;
	P[2] = i2 ;

	return permute(P) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(int i0, int i1, int i2, int i3)
{
	int P[4] ;
	P[0] = i0 ;
	P[1] = i1 ;
	P[2] = i2 ;
	P[3] = i3 ;

	return permute(P) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(int i0, int i1, int i2, int i3, int i4)
{
	int P[5] ;
	P[0] = i0 ;
	P[1] = i1 ;
	P[2] = i2 ;
	P[3] = i3 ;
	P[4] = i4 ;

	return permute(P) ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(int i0, int i1, int i2, int i3, int i4, int i5)
{
	int P[6] ;
	P[0] = i0 ;
	P[1] = i1 ;
	P[2] = i2 ;
	P[3] = i3 ;
	P[4] = i4 ;
	P[5] = i5 ;

	return permute(P) ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(int i0, int i1, int i2, int i3, int i4, int i5, int i6)
{
	int P[7] ;
	P[0] = i0 ;
	P[1] = i1 ;
	P[2] = i2 ;
	P[3] = i3 ;
	P[4] = i4 ;
	P[5] = i5 ;
	P[6] = i6 ;

	return permute(P) ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
	int P[8] ;
	P[0] = i0 ;
	P[1] = i1 ;
	P[2] = i2 ;
	P[3] = i3 ;
	P[4] = i4 ;
	P[5] = i5 ;
	P[6] = i6 ;
	P[7] = i7 ;

	return permute(P) ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8)
{
	int P[9] ;
	P[0] = i0 ;
	P[1] = i1 ;
	P[2] = i2 ;
	P[3] = i3 ;
	P[4] = i4 ;
	P[5] = i5 ;
	P[6] = i6 ;
	P[7] = i7 ;
	P[8] = i8 ;

	return permute(P) ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(Tensor<int> &P)
{
	if (P.min() < 0)
	{
		std::cout << "permute error: permutation indices must be non-negative integers!" << std::endl ;
		exit(0) ;
	}

	if (P.numel() != R)
	{
		std::cout << "permute error: array dimension and the extent of ORDER must be the same! " << std::endl ;
		exit (0) ;
	}
	//-------------------------------------------------
	return permute(P.ValArr()) ;

}

template <typename C>
Tensor<C> Tensor<C>::permute(const int * P)
{
	std::vector<int> dim(R) ;
	std::vector<Index> inds ;
	for (int i = 0; i < R; i++)
	{
		dim[i] = D[ P[i] ] ;

		if (_inds.size() == R)
		{
			inds.push_back(_inds[ P[i] ]) ;
		}
	}
	Tensor<C> A(dim) ;
	if (inds.size() == R)
	{
		A.putIndex(inds) ;
	}


	int n ;
#pragma omp parallel for
	for (n = 0; n < N; n ++)
	{
		int ind[20] ;

		vecInd_to_NDind(n, ind) ;

		int indA[20] ;
		for ( int i = 0; i < R; i ++ )
		{
			indA[i] = ind[ P[i] ] ;
		}

		A(indA) = V[n] ;
	}

	return A ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(const Index& x0, const Index& x1)
{
	int P[2] ;
	std::vector<Index> inds_new = vec(x0, x1) ;

	findIndOrd(inds_new, P) ;

	return permute(P) ;
}

template <typename C>
Tensor<C> Tensor<C>::permute(const Index& x0, const Index& x1, const Index& x2)
{
	int P[3] ;
	std::vector<Index> inds_new = vec(x0, x1, x2) ;

	findIndOrd(inds_new, P) ;

	return permute(P) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3)
{
	int P[4] ;
	std::vector<Index> inds_new = vec(x0, x1, x2, x3) ;

	findIndOrd(inds_new, P) ;

	return permute(P) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4)
{
	int P[5] ;
	std::vector<Index> inds_new = vec(x0, x1, x2, x3, x4) ;

	findIndOrd(inds_new, P) ;

	return permute(P) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4, const Index& x5)
{
	int P[6] ;
	std::vector<Index> inds_new = vec(x0, x1, x2, x3, x4, x5) ;

	findIndOrd(inds_new, P) ;

	return permute(P) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4, const Index& x5, const Index& x6)
{
	int P[7] ;
	std::vector<Index> inds_new = vec(x0, x1, x2, x3, x4, x5, x6) ;

	findIndOrd(inds_new, P) ;

	return permute(P) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4,
														 const Index& x5, const Index& x6, const Index& x7)
{
	int P[8] ;
	std::vector<Index> inds_new = vec(x0, x1, x2, x3, x4, x5, x6, x7) ;

	findIndOrd(inds_new, P) ;

	return permute(P) ;
}
template <typename C>
Tensor<C> Tensor<C>::permute(const Index& x0, const Index& x1, const Index& x2, const Index& x3, const Index& x4,
		              					 const Index& x5, const Index& x6, const Index& x7, const Index& x8)
{
	int P[9] ;
	std::vector<Index> inds_new = vec(x0, x1, x2, x3, x4, x5, x6, x7, x8) ;

	findIndOrd(inds_new, P) ;

	return permute(P) ;
}

template <typename C>
void Tensor<C>::findIndOrd(const std::vector<Index>& inds_new, int * const P)
{
	if ( inds_new.size() != R )
	{
		std::cout << "findIndOrd(std::vector<Index>&, int *) error: number of defined Index must be the same as tensor rank" << std::endl ;
		exit(0) ;
	}
	else if ( _inds.size() != R )
	{
		std::cout << "findIndOrd(std::vector<Index>&, int *) error: number of input Index must be the same as tensor rank" << std::endl ;
		exit(0) ;
	}

	for ( int i = 0; i < R; i ++ ) // loop for P
	{
		for (int j = 0; j < R; j ++) // loop for _inds
		{
			if ( _inds[j] == inds_new[i] )
			{
//				std::cout << j << std::endl ;
				P[i] = j ;
				break ;
			}
		}
	}
}

// Reshape tensor: Index information is lost.

template <typename C>
Tensor<C> Tensor<C>::reshape(int i0)
{

	int num = i0 ;
	if (num != N)
	{
		std::cout << "reshape error: the number of elements must not change." << std::endl ;
		exit(0) ;
	}

	Tensor<C> T(i0) ;

	copyValue(T.ValArr(), V, num) ;

	return T ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(int i0, int i1)
{
	Tensor<int> dim(2) ;
	dim(0) = i0 ;
	dim(1) = i1 ;

	return reshape(dim) ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(int i0, int i1, int i2)
{
	Tensor<int> dim(3) ;
	dim(0) = i0 ;
	dim(1) = i1 ;
	dim(2) = i2 ;

	return reshape(dim) ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(int i0, int i1, int i2, int i3)
{
	Tensor<int> dim(4) ;
	dim(0) = i0 ;
	dim(1) = i1 ;
	dim(2) = i2 ;
	dim(3) = i3 ;

	return reshape(dim) ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(int i0, int i1, int i2, int i3, int i4)
{
	Tensor<int> dim(5) ;
	dim(0) = i0 ;
	dim(1) = i1 ;
	dim(2) = i2 ;
	dim(3) = i3 ;
	dim(4) = i4 ;

	return reshape(dim) ;
}

template <typename C>
Tensor<C> Tensor<C>::reshape(Tensor<int> dim)
{
	int num = dim.prod() ;
	if (num != N)
	{
		std::cout << "reshape error: the number of elements must not change." << std::endl ;
		exit(0) ;
	}
	Tensor<C> T(dim, 0) ;

	copyValue(T.ValArr(), V, num) ;

	return T ;
}
//-- Index information is stored.
template <typename C>
Tensor<C> Tensor<C>::reshape(Index x0)
{
	Tensor<C> T = reshape(x0.dim()) ;
	T.putIndex(x0) ;
	return reshape(x0.dim()) ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(Index x0, Index x1)
{
	Tensor<C> T = reshape(x0.dim(), x1.dim()) ;
	T.putIndex(x0, x1) ;
	return T ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(Index x0, Index x1, Index x2)
{
	Tensor<C> T = reshape(x0.dim(), x1.dim(), x2.dim()) ;
	T.putIndex(x0, x1, x2) ;
	return T ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(Index x0, Index x1, Index x2, Index x3)
{
	Tensor<C> T = reshape(x0.dim(), x1.dim(), x2.dim(), x3.dim()) ;
	T.putIndex(x0, x1, x2, x3) ;
	return T ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(Index x0, Index x1, Index x2, Index x3, Index x4)
{
	Tensor<C> T = reshape(x0.dim(), x1.dim(), x2.dim(), x3.dim(), x4.dim()) ;
	T.putIndex(x0, x1, x2, x3, x4) ;
	return T ;
}
template <typename C>
Tensor<C> Tensor<C>::reshape(std::vector<Index> X)
{
	Tensor<int> dims(X.size()) ;
	for ( int i = 0 ; i < X.size(); i ++ )
	{
		dims.at(i) = X.at(i).dim() ;
	}

	Tensor<C> T = reshape(dims) ;
	T.putIndex(X) ;

	return T ;
}

//-----------------------------------------------------------------------

Tensor<double> Tensor<double>::exp()
{
	Tensor<double> expV = (*this) ;

//	vdExp(N, V, &(expV[1])) ;

	for( int i = 0; i < N; i ++ )
	{
		expV[i] = std::exp(V[i]) ;
	}

	return expV ;
}

// Computes a square root of vector elements.
Tensor<double> Tensor<double>::sqrt()
{
	Tensor<double> sqrtV = (*this) ;

	for ( int i = 0; i < N; i ++)
	{
		sqrtV[i] = std::sqrt(V[i]) ;
	}

	//	vdSqrt(N, V, &(sqrtV[1])) ;

	return sqrtV ;
}

// Raises each element of a vector to the constant power.
Tensor<double> Tensor<double>::power(double exponent)
{
	Tensor<double> powV = (*this) ;

	for ( int i = 0; i < N; i ++)
	{
		powV[i] = std::pow(V[i], exponent) ;
	}

	//	vdPowx(N, V, exponent, &(powV[1])) ;

	return powV ;
}

template <typename C>
C Tensor<C>::mean()
{
	if ( N == 0 )
	{
		std::cout << "Tensor mean error: num of elements is zero" << std::endl ;
		exit(0) ;
	}
	C x = (*this).sum() ;
	x = x / (C)N ;

	return x ;
}

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

template <typename C>
C Tensor<C>::max()
{
	C x = V[0] ;
	for (int i = 1; i < N; i ++)
	{
		if (x < V[i])
		{
			x = V[i] ;
		}
	}
	return x ;
}
// sum of all tensor elements
template <typename C>
C Tensor<C>::sum()
{
	C x = V[0] ;
	for (int i = 1; i < N; i ++)
	{
		x = x + V[i] ;
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
		case '2': // Computes the Euclidean norm of a vector.
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

// return the absolute value of element with maximum absolute value.
double Tensor<double>::maxAbs()
{
	int incx = 1 ;

	int i = idamax(&N, V, &incx) ;

	return fabs(V[i-1]) ;
}

// Sorts numbers in increasing or decreasing order.
// id = 'I': sort d in increasing order;
//    = 'D': sort d in decreasing order.
Tensor<double> Tensor<double>::sort(char id)
{
	if ((D[0] == 1 || D[1] == 1) && R == 2) // row vector || column vector
	{
		Tensor<double> A = *this ;

		int n = A.numel() ; //
		int info = - 1 ;

		dlasrt(&id, &n, &(A[0]), &info) ;

		return A ;
	}
	else
	{
		std::cout << "sort error: it is not a vector." << std::endl ;
		exit(0) ;
	}
}

template <typename C>
void Tensor<C>::symmetrize()
{
	if (R != 2)
	{
		std::cout << "symmetrize error: this must be a matrix." << std::endl ;
		exit(0) ;
	}
	else if ( D[0] != D[1] )
	{
		std::cout << "symmetrize error: this must be a square matrix." << std::endl ;
		exit(0) ;
	}

	for (int i = 0; i < D[0]; i ++)
	{
		for (int j = (i + 1) ; j < D[1]; j ++) //
		{
			C x = ( (*this)(i, j) + (*this)(j, i) ) / 2.0 ;
			(*this)(i, j) = x ;
			(*this)(j, i) = x ;
		}
	}
}

void Tensor<double>::setSmalltoZero(double a)
{
//	#pragma loop count min(512)
	for ( int i = 0; i < N; i ++)
	{
		if ( fabs(V[i]) < a)
		{
			V[i] = 0 ;
		}
	}
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
	return D[i] ;
}

//--------------- dealing with index-----------------
template <typename C>
void Tensor<C>::putIndex(std::vector<Index> &a)
{
//	std::cout << a.size() << std::endl ;
//	std::cout << D[0] << ", " << D[1] << std::endl ;
//	std::cout << "R = " << R << std::endl ;
	if (a.size() == R)
	{
		for (int i = 0; i < R; i ++)
		{
			a.at(i).put_dim(D[i]) ;
		}
		_inds = a ;

	}
	else if ((D[0] == 1 || D[1] == 1) && R == 2 && ( a.size() == 1 ) ) // vector
	{
		a.at(0).put_dim(N) ;
		_inds = a ;
	}
	else if ( ( D[0] == 1 && D[1] == 1 ) && R == 2 && ( a.size() == 0 ) ) // scalar
	{
		_inds = a ;
	}
	else if ( a.size() == 0 ) // empty Index
	{
		return ;
	}
	else
	{
		std::cout << "putIndex(vector<Index>) error: number of Index and tensor rank mismatch." << std::endl ;
		exit(0) ;
	}

}

template <typename C>
void Tensor<C>::putIndex(Index &x0)
{
	if ((D[0] == 1 || D[1] == 1) && R == 2) // row vector || column vector
	{
		_inds.resize(1) ;

		x0.put_dim(N) ;
		_inds[0] = x0 ;

	}
	else
	{
		std::cout << "putIndex(Index) error: Number of Index and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
void Tensor<C>::putIndex(Index &x0, Index &x1)
{
	if ( R == 2 )
	{
		_inds.resize(2) ;

		x0.put_dim(D[0]) ;
		_inds[0] = x0 ;

		x1.put_dim(D[1]) ;
		_inds[1] = x1 ;
	}
	else
	{
		std::cout << "putIndex(Index,Index) error: Number of Index and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
void Tensor<C>::putIndex(Index &x0, Index &x1, Index &x2)
{
	if ( R == 3 )
	{
		_inds.resize(3) ;

		x0.put_dim(D[0]) ;
		_inds[0] = x0 ;

		x1.put_dim(D[1]) ;
		_inds[1] = x1 ;

		x2.put_dim(D[2]) ;
		_inds[2] = x2 ;
	}
	else
	{
		std::cout << "putIndex(Index,Index,Index) error: "
				"Number of Index and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
void Tensor<C>::putIndex(Index &x0, Index &x1, Index &x2, Index &x3)
{
	if ( R == 4 )
	{
		_inds.resize(4) ;

		x0.put_dim(D[0]) ;
		_inds[0] = x0 ;

		x1.put_dim(D[1]) ;
		_inds[1] = x1 ;

		x2.put_dim(D[2]) ;
		_inds[2] = x2 ;

		x3.put_dim(D[3]) ;
		_inds[3] = x3 ;
	}
	else
	{
		std::cout << "putIndex(Index,Index,Index,Index) error: "
				"Number of Index and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
void Tensor<C>::putIndex(Index &x0, Index &x1, Index &x2, Index &x3, Index &x4)
{
	if ( R == 5 )
	{
		_inds.resize(5) ;

		x0.put_dim(D[0]) ;
		_inds[0] = x0 ;

		x1.put_dim(D[1]) ;
		_inds[1] = x1 ;

		x2.put_dim(D[2]) ;
		_inds[2] = x2 ;

		x3.put_dim(D[3]) ;
		_inds[3] = x3 ;

		x4.put_dim(D[4]) ;
		_inds[4] = x4 ;
	}
	else
	{
		std::cout << "putIndex(Index,Index,Index,Index,Index) error: "
				"Number of Index and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
void Tensor<C>::putIndex(Index &x0, Index &x1, Index &x2, Index &x3, Index &x4, Index &x5)
{
	if ( R == 6 )
	{
		_inds.resize(6) ;

		x0.put_dim(D[0]) ;
		_inds[0] = x0 ;

		x1.put_dim(D[1]) ;
		_inds[1] = x1 ;

		x2.put_dim(D[2]) ;
		_inds[2] = x2 ;

		x3.put_dim(D[3]) ;
		_inds[3] = x3 ;

		x4.put_dim(D[4]) ;
		_inds[4] = x4 ;

		x5.put_dim(D[5]) ;
		_inds[5] = x5 ;
	}
	else
	{
		std::cout << "putIndex(Index,Index,Index,Index,Index,Index) error: "
				"Number of Index and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
void Tensor<C>::putIndex(Index &x0, Index &x1, Index &x2, Index &x3, Index &x4, Index &x5, Index &x6)
{
	if ( R == 7 )
	{
		_inds.resize(7) ;

		x0.put_dim(D[0]) ;
		_inds[0] = x0 ;

		x1.put_dim(D[1]) ;
		_inds[1] = x1 ;

		x2.put_dim(D[2]) ;
		_inds[2] = x2 ;

		x3.put_dim(D[3]) ;
		_inds[3] = x3 ;

		x4.put_dim(D[4]) ;
		_inds[4] = x4 ;

		x5.put_dim(D[5]) ;
		_inds[5] = x5 ;

		x6.put_dim(D[6]) ;
		_inds[6] = x6 ;
	}
	else
	{
		std::cout << "putIndex(Index,Index,Index,Index,Index,Index,Index) error: "
				"Number of Index and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}

template <typename C>
void Tensor<C>::putIndex(Index &x0, Index &x1, Index &x2, Index &x3, Index &x4, Index &x5, Index &x6, Index &x7)
{
	if ( R == 8 )
	{
		_inds.resize(8) ;

		x0.put_dim(D[0]) ;
		_inds[0] = x0 ;

		x1.put_dim(D[1]) ;
		_inds[1] = x1 ;

		x2.put_dim(D[2]) ;
		_inds[2] = x2 ;

		x3.put_dim(D[3]) ;
		_inds[3] = x3 ;

		x4.put_dim(D[4]) ;
		_inds[4] = x4 ;

		x5.put_dim(D[5]) ;
		_inds[5] = x5 ;

		x6.put_dim(D[6]) ;
		_inds[6] = x6 ;

		x7.put_dim(D[7]) ;
		_inds[7] = x7 ;
	}
	else
	{
		std::cout << "putIndex(Index,Index,Index,Index,Index,Index,Index,Index) error: "
				"Number of Index and tensor rank mismatch!" << std::endl ;
		exit(0) ;
	}
}
//---------------------------------------------------

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
	info() ;
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
	std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<< std::endl ;
}

template <typename C>
void Tensor<C>::display(int precision)
{
	if (R > 0)
	{
		if (R == 2)
		{
			//			std::cout << "display : " << std::endl ;
			printMatrix(precision) ;
		}
	}
	else
	{
		std::cout << "this is an empty tensor" << std::endl ;
	}
	std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<< std::endl ;
}

template <typename C>
void Tensor<C>::display1D()
{
	info() ;
	for (int i = 0; i < N; i ++ )
	{
		std::cout << V[i] ;
		std::cout << ", " ;
	}
	std::cout << std::endl ;
}

template <typename C>
void Tensor<C>::printMatrix()
{
	for (int i = 0; i < D[0]; i ++)
	{
		//		std::cout << "i = " << i << std::endl ;
		for (int j = 0; j < D[1]; j ++)
		{
			//			std::cout << "j = " << j << std::endl ;
			std::cout << std::setw(14) << std::setprecision(7) << (*this)(i, j) << ',' ;
		}
		std::cout << std::endl ;
	}
}

template <typename C>
void Tensor<C>::printMatrix(int precision)
{
	int wid = precision + 4 ;
	for (int i = 0; i < D[0]; i ++)
	{
		//		std::cout << "i = " << i << std::endl ;
		for (int j = 0; j < D[1]; j ++)
		{
			//			std::cout << "j = " << j << std::endl ;
			std::cout << std::setw(wid) << std::setprecision(precision) << (*this)(i, j) << ',' ;
		}
		std::cout << std::endl ;
	}
}

// output tensor information
template <typename C>
void Tensor<C>::info()
{

	if (typeid(V) == typeid(int*))
	{
		std::cout << "tensor data type: int" << std::endl ;
	}
	else if ( typeid(V) == typeid(double*) )
	{
		std::cout << "tensor data type: double" << std::endl ;
	}

	std::cout << "tensor rank: " << R << std::endl ;
	std::cout << "tensor dimensions: [ " ;
	for (int i = 0; i < R; i ++)
	{
		std::cout << D[i] ;
		if (i < (R - 1))
		{
			std::cout << ", " ;
		}
	}
	std::cout << " ]"<< std::endl ;
	if (_inds.size() == R)
	{
		std::cout << "tensor Index: [" ;
		for ( int i = 0; i < R; i ++ )
		{
			std::cout << _inds[i].ind() ;
			if ( i < (R - 1) )
			{
				std::cout << ", " ;
			}
		}
		std::cout << " ]" << std::endl ;
	}
	else if ( (D[0] == 1 || D[1] == 1) && R == 2 && ( _inds.size() == 1 )  )
	{
		std::cout << "tensor Index: [ " << _inds[0].ind() << " ]" << std::endl ;
	}
	else
	{
		std::cout << "Index is undefined." << std::endl ;
	}
	std::cout << "Total number of elements: " << N << std::endl ;
	std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<< std::endl ;

}
//-------------------------------------------------------------------------------------------

// save/load tensor in binary mode
// 'fileName' includes path
template <typename C>
void Tensor<C>::save(std::string fileName)
{
	std::ofstream outfile(&(fileName[0]), std::ios::binary) ;

	if (!outfile)
	{
		std::cerr<<"save tensor open error!" << std::endl ;
		exit(0) ;
	}

	outfile.write((char*)&R, sizeof(int)) ;
	outfile.write((char*)&N, sizeof(int)) ;
	if ( R != -1 )
	{
		outfile.write((char*)D, sizeof(int) * R) ;
		outfile.write((char*)V, sizeof(C) * N) ;
	}
	outfile.close() ;
}

template <typename C>
void Tensor<C>::load(std::string fileName)
{
	std::ifstream infile(&(fileName[0]), std::ios::binary) ;

	if (!infile)
	{
		std::cerr << "load tensor open error" << std::endl ;
		exit(0) ;
	}
	//***************************************************
	// if original tensor is not empty, delete D
	if (R > 0)
	{
		delete[] D ;
	}
	infile.read((char*)&R, sizeof(int)) ;
	// //////////////////////////////////////////////////
	// if original tensor is not empty, delete V
	if (N > 0)
	{
		delete[] V ;
	}
	infile.read((char*)&N, sizeof(int)) ;
	// //////////////////////////////////////////////////
	// if loaded tensor is not empty, create D
	if (R > 0)
	{
		D = new int[R] ;

		infile.read((char*)D, sizeof(int) * R) ;
	}
	else // if loaded tensor is empty
	{
		D = NULL ;
	}
	// //////////////////////////////////////////////////
	// if loaded tensor is not empty, create V
	if (N > 0)
	{
		V = new C[ N ] ;
		infile.read((char*)V, sizeof(C) * N) ;
	}
	else // if loaded tensor is empty
	{
		V = NULL ;
	}
	//***************************************************
	infile.close() ;
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// non-member functions

// overload operators
template <typename C>
Tensor<C> operator + (Tensor<C>& A, Tensor<C>& B)
{

	if (isAllDimAgree(A, B)) // check if A and B are tensors with same size(rank and dim)
	{
		Tensor<C> T = A ;
		for (int i = 0; i < A.numel(); i++)
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
	for (int i = 0; i < B.numel(); i++)
	{
		T[i] = a + B[i] ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator + (Tensor<C>& A, C b)
{
	Tensor<C> T = A ;
	for ( int i = 0; i < A.numel(); i++)
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
		for (int i = 0; i < A.numel(); i++)
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
	for (int i = 0; i < B.numel(); i++)
	{
		T[i] = a - B[i] ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator - (Tensor<C>& A, C b)
{
	Tensor<C> T = A ;
	for ( int i = 0; i < A.numel(); i++)
	{
		T[i] = A[i] - b ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator * (C a, Tensor<C>& B)
{
	Tensor<C> T = B ;
	for (int i = 0; i < B.numel(); i++)
	{
		T[i] = a * B[i] ;
	}
	return T ;
}

template <typename C>
Tensor<C> operator * (Tensor<C>& A, C b)
{
	Tensor<C> T = A ;
	for ( int i = 0; i < A.numel(); i++)
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

	int m = A.dimension(0) ;
	int k = A.dimension(1) ;

	int ldB = B.dimension(0) ;
	int n = B.dimension(1) ;

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
	dgemm(&transa, &transb, &m, &n, &k, &alpha, &(A[0]), &m, &(B[0]), &k, &beta, &(AB[0]), &m) ;

	return AB ;
}

template <typename C>
Tensor<C> operator / (Tensor<C>& A, C b)
{
	Tensor<C> T = A ;
	for ( int i = 0; i < A.numel(); i++)
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
	for (int i = 0; i < A.rank(); i++)
	{
		if (A.dimension(i) != B.dimension(i))
		{
			return false ;
		}
	}
	return true ;
}

double compareTensors(Tensor<double>& A, Tensor<double>& A0)
{
	if (!isAllDimAgree(A, A0))
	{
		std::cout << "compareTensors error: input two tensors are not the same dimension." << std::endl ;
		std::cout << "they are not comparable." << std::endl ;
		exit(0) ;
	}
	else
	{
		Tensor<double> Adiff = A - A0 ;
		double c = Adiff.maxAbs() / A.maxAbs() ;
		return c ;
	}
}

// compute dot product of A and B as vectors
template <typename C>
C dot(Tensor<C> &A, Tensor<C> &B)
{
	int na = A.numel() ;
	int nb = B.numel() ;

	if (na != nb)
	{
		std::cout << "dot error: number of elements must agree." << std::endl ;
		exit(0) ;
	}
	else
	{
		C x = 0 ;
		for (int i = 0; i < na; i ++)
		{
			x = x + A[i] * B[i] ;
		}
		return x ;
	}
}

// contract same Index of the 2 tensors
Tensor<double> contractTensors(Tensor<double>& A, Tensor<double>& B)
{
	if (A.inds().size() != A.rank() || B.inds().size() != B.rank())
	{
		std::cout << "contractTensors(Tensor<double>,Tensor<double>) error: tensor Index must be defined with correct length" << std::endl ;
		exit(0) ;
	}

	Tensor<int> indA_ord_right, indB_ord_left ;
	std::vector<Index> indsAB = findCommonIndex(A.inds(), indA_ord_right, B.inds(), indB_ord_left) ;
//	std::cout << indsAB.size() << std::endl ;
	Tensor<double> AB = contractTensors(A, indA_ord_right, B, indB_ord_left) ;
	AB.putIndex(indsAB) ;

	return AB ;

}

std::vector<Index> findCommonIndex(std::vector<Index> indsA, Tensor<int> &indA_ord_right,
		                               std::vector<Index> indsB, Tensor<int> &indB_ord_left)
{
	std::vector<int> ordA, ordB ;
	std::vector<Index> indsCommon ;

	for (int i = 0; i < indsA.size(); i ++)
	{
		for ( int j = 0; j < indsB.size(); j ++ )
		{
			if ( indsA[i] == indsB[j] )
			{
				ordA.push_back(i) ;
				ordB.push_back(j) ;
				indsCommon.push_back(indsA[i]) ;
				break ;
			}
		}
	}

	indA_ord_right = Tensor<int>(ordA.size()) ;
	indB_ord_left = Tensor<int>(ordB.size()) ;

	for (int i = 0; i < ordA.size(); i ++)
	{
		indA_ord_right[i] = ordA.at(i) ;
		indB_ord_left[i] = ordB.at(i) ;
	}
	//---------------------------------------------
	std::vector<Index> indsAB ;
	for ( int i = 0; i < indsA.size(); i ++ )
	{
		bool existence = false ;
		for ( int j = 0; j < indsCommon.size(); j ++ )
		{
			if (indsA.at(i) == indsCommon.at(j) )
			{
				existence = true ;
				break ;
			}
		}
		if ( !existence )
		{
//			std::cout << indsA.at(i).ind() << std::endl ;
//			exit(0) ;
			indsAB.push_back(indsA.at(i)) ;
		}
	}

	for ( int i = 0; i < indsB.size(); i ++ )
	{
		bool existence = false ;
		for ( int j = 0; j < indsCommon.size(); j ++ )
		{
			if ( indsB.at(i) == indsCommon.at(j) )
			{
				existence = true ;
				break ;
			}
		}
		if ( !existence )
		{
			indsAB.push_back(indsB.at(i)) ;
		}
	}

//	std::cout << indsAB.at(0).ind() << std::endl ;
//	exit(0) ;
	return indsAB ;

}

Tensor<double> contractTensors(Tensor<double>& A, int a0, Tensor<double>& B, int b0)
{
	Tensor<int> indA_right(1) ;
	indA_right(0) = a0 ;

	Tensor<int> indB_left(1) ;
	indB_left(0) = b0 ;

	return contractTensors(A, indA_right, B, indB_left) ;

}

Tensor<double> contractTensors(Tensor<double>& A, int a0, int a1, Tensor<double>& B, int b0, int b1)
{
	Tensor<int> indA_right(2) ;
	indA_right(0) = a0 ;
	indA_right(1) = a1 ;

	Tensor<int> indB_left(2) ;
	indB_left(0) = b0 ;
	indB_left(1) = b1 ;

	return contractTensors(A, indA_right, B, indB_left) ;
}

Tensor<double> contractTensors(Tensor<double>& A, int a0, int a1, int a2, Tensor<double>& B, int b0, int b1, int b2)
		{
	Tensor<int> indA_right(3) ;
	indA_right(0) = a0 ;
	indA_right(1) = a1 ;
	indA_right(2) = a2 ;

	Tensor<int> indB_left(3) ;
	indB_left(0) = b0 ;
	indB_left(1) = b1 ;
	indB_left(2) = b2 ;

	return contractTensors(A, indA_right, B, indB_left) ;

		}

Tensor<double> contractTensors(Tensor<double>& A, int a0, int a1, int a2, int a3,
		Tensor<double>& B, int b0, int b1, int b2, int b3)
		{
	Tensor<int> indA_right(4) ;
	indA_right(0) = a0 ;
	indA_right(1) = a1 ;
	indA_right(2) = a2 ;
	indA_right(3) = a3 ;

	Tensor<int> indB_left(4) ;
	indB_left(0) = b0 ;
	indB_left(1) = b1 ;
	indB_left(2) = b2 ;
	indB_left(3) = b3 ;

	return contractTensors(A, indA_right, B, indB_left) ;

		}

Tensor<double> contractTensors(Tensor<double> &A0, Tensor<int> indA_right, Tensor<double> &B0, Tensor<int> indB_left)
{
	double t_start, t_end ;

//	t_start = dsecnd() ;
	// A(indA_left, indA_right)
	// if indA_right.numel() == A.rank(), indA_left is an empty tensor
	Tensor<int> indA_left = findOuterInd(A0.rank(), indA_right) ;
	// B(indB_left, indB_right)
	// if indB_left.numel() == B.rank(), indB_right is an empty tensor
	Tensor<int> indB_right = findOuterInd(B0.rank(), indB_left) ;
//	t_end = dsecnd() ;
//	std::cout << "findOuterInd time cost: " << (t_end - t_start) << " s" << std::endl ;
//	t_start = dsecnd() ;
	Tensor<int> indA = vertcat(indA_left, indA_right) ;
	Tensor<int> indB = vertcat(indB_left, indB_right) ;
//	t_end = dsecnd() ;
//	std::cout << "vertcat time cost: " << (t_end - t_start) << " s" << std::endl ;
	//-----------------------------------------------------------------------------
//	t_start = dsecnd() ;
	Tensor<double> A = A0.permute(indA) ;
	Tensor<double> B = B0.permute(indB) ;
//	t_end = dsecnd() ;
//	std::cout << "permute time cost: " << (t_end - t_start) << " s" << std::endl ;
	//------------------------------------------------------------------------------
//	t_start = dsecnd() ;
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

	int id = 0 ; // iteration to find dim of AB
	// compute matrix product
	int m = 1 ; // left dimension of A
	// if indA_left is an empty tensor, m = 1
	for (int i = 0; i < indA_left.numel(); i ++)
	{
		m = m * A.dimension(i) ;
		dimAB(id) = A.dimension(i) ;
		id ++ ;
	}

	if (id == 0 && rankAB < 2)
	{
		id ++ ;
	}

	int k = 1 ; // right dimension A
	for(int i = indA_left.numel(); i < A.rank() ; i ++)
	{
		k = k * A.dimension(i) ;
	}

	int ldB = 1 ; // left dimension of B
	for (int i = 0; i < indB_left.numel(); i ++)
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
	for (int i = indB_left.numel(); i < B.rank(); i ++)
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
//	t_end = dsecnd() ;
//	std::cout << "define dgemm parameter time cost: " << (t_end - t_start) << " s" << std::endl ;
//	std::cout << "contractTensors check point" << std::endl ;

//	t_start = dsecnd() ;
	dgemm(&transa, &transb, &m, &n, &k, &alpha, &(A[0]), &m, &(B[0]), &k, &beta, &(AB[0]), &m) ;
//	t_end = dsecnd() ;
//	std::cout << "dgemm time cost: " << (t_end - t_start) << " s" << std::endl ;

	return AB ;
}

Tensor<int> findOuterInd(int rank, Tensor<int> ind_inner)
{
	bool existence = false ;
	int k = 0 ;
	Tensor<int> ind_outer(rank - ind_inner.numel()) ;

	for (int i = 0; i < rank; i ++) // iteration for all indices
	{
		for (int j = 0; j < ind_inner.numel(); j ++ )
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

	int j = 0 ;
	for (int i = 0; i < ind_left.numel(); i ++)
	{

		ind(j) = ind_left(i) ;
		j ++ ;
	}
	for (int i = 0; i < ind_right.numel(); i ++)
	{
		ind(j) = ind_right(i) ;
		j ++ ;
	}
	return ind ;
}

Tensor<double> absorbVector(Tensor<double>& A, int n, Tensor<double>& Vec)
{
	Tensor<double> AV = A ;

#pragma omp parallel for
	for (int i = 0; i < A.numel(); i ++)
	{
		int ind[20] ;
		A.vecInd_to_NDind(i, ind) ;

		AV[i] = A[i] * Vec(ind[n]) ;
	}

	return AV ;
	//-----------------------------------------------------------------------
//	int r = A.rank() ;
//
//	if (r == 2)
//	{
//		return tensor2xVec(A, n, Vec) ;
//	}
//	else if (r == 3)
//	{
//		return tensor3xVec(A, n, Vec) ;
//	}
//	else if (r == 4)
//	{
//		return tensor4xVec(A, n, Vec) ;
//	}
//	else if (r == 5)
//	{
//		return tensor5xVec(A, n, Vec) ;
//	}
//	else
//	{
//		std::cout << "absorbVector error: can not deal with tensor rank: " << r << std::endl ;
//		exit(0) ;
//	}

}

Tensor<double> spitVector(Tensor<double>& A, int n, Tensor<double>& Vec)
{
	Tensor<double> AV = A ;
#pragma omp parallel for
	for (int i = 0; i < A.numel(); i ++)
	{
		int ind[20] ;
		A.vecInd_to_NDind(i, ind) ;
		AV[i] = A[i] / Vec(ind[n]) ;
	}
	return AV ;

	//-----------------------------------------------------------
//	Tensor<double> invV = Vec ;
//	for(int i = 0; i < Vec.numel(); i ++)
//	{
//		invV(i) = 1.0 / Vec(i) ;
//	}
//	int r = A.rank() ;
//	//
//	if (r == 2)
//	{
//		return tensor2xVec(A, n, invV) ;
//	}
//	else if (r == 3)
//	{
//		return tensor3xVec(A, n, invV) ;
//	}
//	else if (r == 4)
//	{
//		return tensor4xVec(A, n, invV) ;
//	}
//	else if (r == 5)
//	{
//		return tensor5xVec(A, n, invV) ;
//	}
//	else
//	{
//		std::cout << "absorbVector error: can not deal with tensor rank: " << r << std::endl ;
//		exit(0) ;
//	}

}

//Tensor<double> tensor2xVec(Tensor<double>& A, int i, Tensor<double>& Vec)
//{
//	Tensor<double> AV = A ;
//	Tensor<int> ind(2) ;
//
//	int d0 = A.dimension(0) ;
//	int d1 = A.dimension(1) ;
//
//	int j0, j1 ;
//	int n = 0 ;
//	for (j1 = 0; j1 < d1; j1 ++)
//	{
//		ind(1) = j1 ;
//		for (j0 = 0; j0 < d0; j0 ++)
//		{
//			ind(0) = j0 ;
//			//---------------------
//			AV[n] = A[n] * Vec(ind(i)) ;
//			n ++ ;
//		}
//	}
//
//	return AV ;
//}
//Tensor<double> tensor3xVec(Tensor<double>& A, int i, Tensor<double>& V)
//{
//	Tensor<double> AV = A ;
//	Tensor<int> ind(3) ;
//
//	int d1 = A.dimension(1) ;
//	int d2 = A.dimension(2) ;
//	int d3 = A.dimension(3) ;
//
//	int j1, j2, j3 ;
//	int n = 1 ;
//	for (j3 = 1; j3 <= d3; j3 ++)
//	{
//		ind(3) = j3 ;
//		for (j2 = 1; j2 <= d2; j2 ++)
//		{
//			ind(2) = j2 ;
//			for (j1 = 1; j1 <= d1; j1 ++)
//			{
//				ind(1) = j1 ;
//				//---------------------
//				AV[n] = A[n] * V(ind(i)) ;
//				n ++ ;
//			}
//		}
//	}
//	return AV ;
//}
//Tensor<double> tensor4xVec(Tensor<double>& A, int i, Tensor<double>& V)
//{
//	Tensor<double> AV = A ;
//	Tensor<int> ind(4) ;
//
//	int d1 = A.dimension(1) ;
//	int d2 = A.dimension(2) ;
//	int d3 = A.dimension(3) ;
//	int d4 = A.dimension(4) ;
//
//	int j1, j2, j3, j4 ;
//	int n = 1 ;
//	for (j4 = 1; j4 <= d4; j4 ++)
//	{
//		ind(4) = j4 ;
//		for (j3 = 1; j3 <= d3; j3 ++)
//		{
//			ind(3) = j3 ;
//			for (j2 = 1; j2 <= d2; j2 ++)
//			{
//				ind(2) = j2 ;
//				for (j1 = 1; j1 <= d1; j1 ++)
//				{
//					ind(1) = j1 ;
//					//---------------------
//					AV[n] = A[n] * V(ind(i)) ;
//					n ++ ;
//				}
//			}
//		}
//	}
//	return AV ;
//}
//Tensor<double> tensor5xVec(Tensor<double>& A, int i, Tensor<double>& V)
//{
//	Tensor<double> AV = A ;
//	Tensor<int> ind(5) ;
//
//	int d1 = A.dimension(1) ;
//	int d2 = A.dimension(2) ;
//	int d3 = A.dimension(3) ;
//	int d4 = A.dimension(4) ;
//	int d5 = A.dimension(5) ;
//
//	int j1, j2, j3, j4, j5 ;
//	int n = 1 ;
//	for (j5 = 1; j5 <= d5; j5 ++)
//	{
//		ind(5) = j5 ;
//		for (j4 = 1; j4 <= d4; j4 ++)
//		{
//			ind(4) = j4 ;
//			for (j3 = 1; j3 <= d3; j3 ++)
//			{
//				ind(3) = j3 ;
//				for (j2 = 1; j2 <= d2; j2 ++)
//				{
//					ind(2) = j2 ;
//					for (j1 = 1; j1 <= d1; j1 ++)
//					{
//						ind(1) = j1 ;
//						//---------------------
//						AV[n] = A[n] * V(ind(i)) ;
//						n ++ ;
//					}
//				}
//			}
//		}
//	}
//	return AV ;
//}

//-------------------------------------------with Lapack------------------------------------------------

// Computes eigenvalues and eigenvectors of a real symmetric matrix using the Relatively Robust Representations.
// A * Z = Z * W ,
// Z: the columns of Z are the right eigenvectors.
// W: W is a column vector which contains the eigenvalue in increasing order.
int symEig(Tensor<double> A, Tensor<double>& Z, Tensor<double>& W)
{
	if (A.rank() != 2)
	{
		std::cout << "symEig error: 1st input must be a matrix." << std::endl ;
		exit(0) ;
	}
	if (A.dimension(0) != A.dimension(1))
	{
		std::cout << "symEig error: 1st input matrix must be square." << std::endl ;
		exit(0) ;
	}

	char jobz = 'V' ; // eigenvalues and eigenvectors are computed.
	char range = 'A' ; // the routine computes all eigenvalues.
	char uplo = 'U' ; // a stores the upper triangular part of A.

	int n = A.dimension(0) ; // The order of the matrix A.

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

	dsyevr(&jobz, &range, &uplo, &n, &(A[0]), &lda, &vl, &vu, &il, &iu, &abstol, &m,
			&(W[0]), &(Z[0]), &ldz, &(isuppz[0]), &(work[0]), &lwork, &(iwork[0]), &liwork, &info) ;

	//********************************************************************
	// computation
	lwork = (int)work(0) ;
	work = Tensor<double>(lwork) ;
	liwork = iwork(0) ;
	iwork = Tensor<int>(liwork) ;

	dsyevr(&jobz, &range, &uplo, &n, &(A[0]), &lda, &vl, &vu, &il, &iu, &abstol, &m,
			&(W[0]), &(Z[0]), &ldz, &(isuppz[0]), &(work[0]), &lwork, &(iwork[0]), &liwork, &info) ;

	if (info != 0)
	{
		std::cout << "info = " << info << std::endl ;
	}

	return info ;
}

// Computes the eigenvalues and left and right eigenvectors of a real non-symmetric matrix.
// input:
// A (real non-symmetric matrix)
// output:
// Wr: column vector which contains real part of eigenvalues
// Wi: column vector which contains imaginary part of eigenvalues
// Vl: left eigenvectors are stored in the columns, complex conjugate pair are Vl(:,j)+i*Vl(:,j+1) and Vl(:,j)-i*Vl(:,j+1)
// Vr: right eigenvectors are stored in the columns, complex conjugate pair are Vr(:,j)+i*Vr(:,j+1) and Vr(:,j)-i*Vr(:,j+1)
int reNonSymEig(Tensor<double> A, Tensor<double> &Wr, Tensor<double> &Wi, Tensor<double> &Vl, Tensor<double> &Vr)
{
	if (A.rank() != 2)
	{
		std::cout << "reNonSymEig error: 1st input must be a matrix." << std::endl ;
		exit(0) ;
	}
	if (A.dimension(0) != A.dimension(1))
	{
		std::cout << "reNonSymEig error: 1st input matrix must be square." << std::endl ;
		exit(0) ;
	}

	char jobvl = 'V' ; // left eigenvectors of A are computed.
	char jobvr = 'V' ; // right eigenvectors of A are computed.

	int n = A.dimension(0) ; // The order of the matrix A (n �����ゆ�烽����ゆ�烽��濮�锟斤拷������锟芥�ゆ�烽��浠�锟斤拷���锟� 0)

	int lda = n ; // The leading dimension of the array a. Must be at least max(1, n).
	// The leading dimensions of the output arrays vl and vr
	int ldvl = n ;
	int ldvr = n ;
	/*
	 *  Contain the real and imaginary parts, respectively, of the computed eigenvalues.
	 *  Complex conjugate pairs of eigenvalues appear consecutively
	 *  with the eigenvalue having positive imaginary part first.
	 */
	Wr = Tensor<double>(n) ;
	Wi = Tensor<double>(n) ;
	/*
	 *  left eigenvectors are stored in the columns,
	 *  complex conjugate pair are Vl(:,j)+i*Vl(:,j+1) and Vl(:,j)-i*Vl(:,j+1)
	 */
	Vl = Tensor<double>(n, n) ;
	/*
	 *  right eigenvectors are stored in the columns,
	 *  complex conjugate pair are Vr(:,j)+i*Vr(:,j+1) and Vr(:,j)-i*Vr(:,j+1)
	 */
	Vr = Tensor<double>(n, n) ;

	int info ; //
	//***************************************************
	// workspace query
	Tensor<double> work(1) ;
	int lwork = - 1 ;

	dgeev(&jobvl, &jobvr, &n, &(A[0]), &lda, &(Wr[0]), &(Wi[0]), &(Vl[0]), &ldvl, &(Vr[0]), &ldvr, &(work[0]), &lwork, &info) ;

	//***************************************************
	// computation
	lwork = (int)work(0) ;
	work = Tensor<double>(lwork) ;

	dgeev(&jobvl, &jobvr, &n, &(A[0]), &lda, &(Wr[0]), &(Wi[0]), &(Vl[0]), &ldvl, &(Vr[0]), &ldvr, &(work[0]), &lwork, &info) ;

	if (info != 0)
	{
		std::cout << "info = " << info << std::endl ;
	}
	//**************************************************
	return info ;
}

// Computes the singular value decomposition of a general rectangular matrix using a divide and conquer method.
// A = U * S * V'
// info = svd(A, U, S, V) ;
int svd(Tensor<double> A, Tensor<double>& U, Tensor<double>& S, Tensor<double>& V)
{
	if (A.rank() != 2)
	{
		std::cout << "svd error: 1st input must be a matrix." << std::endl ;
		exit(0) ;
	}

	char jobz = 'S' ;

	int m = A.dimension(0) ; // The number of rows of the matrix A
	int n = A.dimension(1) ; // The number of columns in A

	int lda = m ; // The leading dimension of  A
	int ldu = m ; // The leading dimensions of U
	int ldvt = std::min(m, n) ; // The leading dimensions of VT

	S = Tensor<double>(ldvt) ;
	U = Tensor<double>(m, ldvt) ;
	V = Tensor<double>(ldvt, n) ;

	Tensor<int> iwork(8 * ldvt) ;

	int info ;
	//************************************************
	// workspace query
	Tensor<double> work(1) ;
	int lwork = - 1 ;

	dgesdd(&jobz, &m, &n, &(A[0]), &lda, &(S[0]), &(U[0]), &ldu, &(V[0]), &ldvt, &(work[0]), &lwork, &(iwork[0]), &info) ;

	//*********************************************************************
	// computation
	lwork = (int)work(0) ;
	work = Tensor<double>(lwork) ;

	dgesdd(&jobz, &m, &n, &(A[0]), &lda, &(S[0]), &(U[0]), &ldu, &(V[0]), &ldvt, &(work[0]), &lwork, &(iwork[0]), &info) ;

//	double t_start = dsecnd() ;
	V = V.trans() ;
//	double t_end = dsecnd() ;
//	std::cout << "trans() time cost: " << (t_end - t_start) << std::endl ;
	//
	if (info != 0)
	{
		std::cout << "svd error info = " << info << std::endl ;
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

	int m = A.dimension(0) ; // The number of rows in the matrix A
	int n = A.dimension(1) ; // The number of columns in A

	int k = std::min(m, n) ;

	R = A ;

	int lda = m ;

	Tensor<double> tau(k) ;

	int info ;
	//************************************************
	// workspace query
	Tensor<double> work(1) ;
	int lwork = - 1 ;

	dgeqrf(&m, &n, &(R[0]), &lda, &(tau[0]), &(work[0]), &lwork, &info) ;

	//************************************************
	// computation
	lwork = (int)work(0) ;
	work = Tensor<double>(lwork) ;

	dgeqrf(&m, &n, &(R[0]), &lda, &(tau[0]), &(work[0]), &lwork, &info) ;

	if (info != 0)
	{
		std::cout << "info = " << info << std::endl ;
	}

	if ( m > n)
	{
		R = R.subTensor(0,n-1, 0,n-1) ;
	}

	for (int i = 0; i < R.dimension(0); i ++)
	{
		for (int j = 0; j < i ; j ++)
		{
			R(i, j) = 0 ;
		}
	}
	return info ;
}

// LQ factorization
// Computes the LQ factorization of a general m-by-n matrix.
// A = L * Q
int lq(Tensor<double>& A, Tensor<double>& L)
{
	if (A.rank() != 2)
	{
		std::cout << "lq error: 1st input must be a matrix." << std::endl ;
		exit(0) ;
	}

	int m = A.dimension(0) ; // The number of rows in the matrix A
	int n = A.dimension(1) ; // The number of columns in A

	int k = std::min(m, n) ;

	L = A ;

	int lda = m ;

	Tensor<double> tau(k) ;

	int info ;
	//************************************************
	// workspace query
	Tensor<double> work(1) ;
	int lwork = - 1 ;

	dgelqf(&m, &n, &(L[0]), &lda, &(tau[0]), &(work[0]), &lwork, &info) ;

	//************************************************
	// computation
	lwork = (int)work(0) ;
	work = Tensor<double>(lwork) ;

	dgelqf(&m, &n, &(L[0]), &lda, &(tau[0]), &(work[0]), &lwork, &info) ;

	if (info != 0)
	{
		std::cout << "info = " << info << std::endl ;
	}

	if ( m < n)
	{
		L = L.subTensor(0,m-1, 0,m-1) ;
	}
	// upper triangular part is set to zero
	for (int i = 0; i < L.dimension(0); i ++)
	{
		for (int j = (i + 1); j < L.dimension(1) ; j ++)
		{
			L(i, j) = 0 ;
		}
	}

	return info ;
}
//***********************************************************************

//---------------------------------------------------------------------------------------------------------
#endif /* TENSOR_HPP_ */
