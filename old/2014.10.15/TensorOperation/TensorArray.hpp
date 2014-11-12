/*
 * TensorArray.hpp
 *
 *  Created on: 2014-9-26
 *  Updated on: 2014-10-2
 *      Author: zhaohuihai
 */

#ifndef TENSORARRAY_HPP_
#define TENSORARRAY_HPP_

#include "Tensor.hpp"

template <typename C>
class TensorArray
{
private:

	// Rank of tensor array
	int R ;
	// Total number of tensors
	int N ;
	// Array of dimension of R indices
	int* D ;
	// array of N tensors
	Tensor<C>* V ;
	//--------------------------------------------------------------------------------------
	Tensor<C>& at(Tensor<int> &position) ;
	bool outOfBound(Tensor<int> &position) ;
	void copy(const TensorArray<C>& A) ;
public:
	// constructor
	TensorArray() ; // empty tensor array
	TensorArray(const TensorArray<C>& A) ; // overload copy constructor
	// create a d0 dimensional column vector, which is represented as a d0 X 1 tensors.
	TensorArray(int d0) ; // all elements are 0
	TensorArray(int d0, int d1) ;
	TensorArray(int d0, int d1, int d2) ;
	//     destructor
	~TensorArray() ;
	//--------------
	void randUniform() ;
	//-----------------------------------------------------------------------------------
	// overload operators
	TensorArray<C>& operator = (const TensorArray<C>& A) ; // copy tensor array
	Tensor<C>& operator [] (const int n) const ; // get a tensor by array index starting from 0
	Tensor<C>& at (const int n) const ; // with bound check
	// Access operator of a tensor in Fortran order. Get a tensor by tensor array index
	Tensor<C>& operator () (const int i0) ;
	Tensor<C>& operator () (const int i0, const int i1) ;
	Tensor<C>& operator () (const int i0, const int i1, const int i2) ;
	Tensor<C>& operator () (Tensor<int> ind) ;
	//------------------------------------------------------------------------------------
	int tensorIndToArrayPos(Tensor<int> ind) ; // convert tensor index to array position
	Tensor<int> arrayPosToTensorInd(int p) ; // convert array position to tensor index
	//------------------------------------------------------------------------------------
	void exchange(int i0, int i1) ;
	//------------------------------------------------------------------------------------

	int rank() const ;
	int numel() const ;
	int* dimArr() const ;
	Tensor<C>* ValArr() const ;
	int dimension(int i) const ;

	void display() ;
	void info() ;
	//------------------------------------------------------------------------------------
	void save(std::string dirName) ;
	void load(std::string dirName) ;
};
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// constructor
template <typename C>
TensorArray<C>::TensorArray()
{
	R = - 1 ;
	D = NULL ;

	N = 0 ;
	V = NULL ;
}
// overload copy constructor
template <typename C>
TensorArray<C>::TensorArray(const TensorArray<C>& A)
{
	R = - 1 ;
	D = NULL ;

	N = 0 ;
	V = NULL ;

	copy(A) ;
}

// create a d0 dimensional column vector, which is represented as a d0 X 1 tensors.
template <typename C>
TensorArray<C>::TensorArray(int d0)
{
	R = 2 ;
	//
	N = d0 ;
	// column vector
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = 1 ; // column vector
	//
	V = new Tensor<C> [N] ;

}

template <typename C>
TensorArray<C>::TensorArray(int d0, int d1)
{
	R = 2 ;
	//
	N = d0 * d1 ;
	//
	D = new int[R] ;
	D[0] = d0 ;
	D[1] = d1 ;
	//
	V = new Tensor<C> [N] ;
}

template <typename C>
TensorArray<C>::TensorArray(int d0, int d1, int d2)
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
	V = new Tensor<C> [N] ;
}
//     destructor
template <typename C>
TensorArray<C>::~TensorArray()
{
	//std::cout << "call destructor of TensorArray" << std::endl ;
	if ( N > 0 )
	{
		delete[] V ;
	}
	if ( R > 0 )
	{
		delete[] D ;
	}
	R = - 1 ;
	D = NULL ;

	N = 0 ;
	V = NULL ;
}

template <typename C>
void TensorArray<C>::copy(const TensorArray<C>& A)
{
	// if lhs of tensor array is not empty, delete D
	if (R > 0)
	{
		delete[] D ;
	}
	R = A.rank() ;
	// if rhs tensor array is not empty, create D
	if ( R > 0)
	{
		D = new int[R] ;
		for (int i = 0; i < R; i ++)
		{
			D[i] = A.dimension(i) ;
		}
	}
	else // if rhs is empty
	{
		D = NULL ;
	}

	// lhs tensor array is not empty, delete V
	if (N > 0)
	{
		delete[] V ;
	}
	N = A.numel() ;
	// if rhs tensor array is not empty, create V
	if ( N > 0 )
	{
		V = new Tensor<C> [N] ;
		for ( int i = 0; i < N; i ++)
		{
			V[i] = A[i] ;
		}
	}
	else // if rhs is empty
	{
		V = NULL ;
	}
}
//---------------------------------------------------------------------------------------------------------
void TensorArray<double>::randUniform()
		{
	for ( int i = 0 ; i < N; i ++)
	{
		V[i].randUniform() ;
	}
		}
//---------------------------------------------------------------------------------------------------------
// overload operators

// copy tensor array
template <typename C>
TensorArray<C>& TensorArray<C>::operator =(const TensorArray<C>& A)
{
	copy(A) ;

	return *this ;
}

template <typename C>
Tensor<C>& TensorArray<C>::operator [](const int n) const
{
	return V[n] ;
}

template <typename C>
Tensor<C>& TensorArray<C>::at(const int n) const
{
	if (n < 0 )
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
//********************************************************************
// overload "()"
template <typename C>
Tensor<C>& TensorArray<C>::operator () (const int i0)
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
Tensor<C>& TensorArray<C>::operator () (const int i0, const int i1)
{

	//++++++++++++++++++++++++++++++++++++
	return V[ i0 + i1 * D[0] ] ;
}

template <typename C>
Tensor<C>& TensorArray<C>::operator () (const int i0, const int i1, const int i2)
{

	//++++++++++++++++++++++++++++++++++++
	return V[ i0 + i1 * D[0] + i2 * D[0] * D[1]] ;
}

template <typename C>
Tensor<C>& TensorArray<C>::operator () (Tensor<int> ind)
{
	return at(ind) ;
}

// private member function
template <typename C>
Tensor<C>& TensorArray<C>::at(Tensor<int> &position)
{
	int p = tensorIndToArrayPos(position) ;
	return V[p] ;
}

template <typename C>
bool TensorArray<C>::outOfBound(Tensor<int> &position)
{
	if (position.numel() == 1)
	{
		if (N <= position.at(0))
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

//---------------------------------------------------------------------------------------------------------
template <typename C>
int TensorArray<C>::tensorIndToArrayPos(Tensor<int> ind) // convert tensor index to array position
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

// convert array position to tensor index
template <typename C>
Tensor<int> TensorArray<C>::arrayPosToTensorInd(int p)
{

	// this is a generalized positional notation
	//--------------------------------------------------
	Tensor<int> ind(R) ;
	int i ;
	for (int i = 0; i < R; i ++)
	{
		ind[i] = p % D[i] ; //
		p /= D[i] ; //
	}
	//----------------------------------------
	return ind ;
}
//---------------------------------------------------------------------------------------------------------

template <typename C>
void TensorArray<C>::exchange(int i0, int i1)
{
	Tensor<C> Temp = (*this)(i0) ;
	(*this)(i0) = (*this)(i1) ;
	(*this)(i1) = Temp ;
}

//---------------------------------------------------------------------------------------------------------
template <typename C>
int TensorArray<C>::rank() const
{
	return R ;
}

template <typename C>
int TensorArray<C>::numel() const
{
	return N ;
}

template <typename C>
int* TensorArray<C>::dimArr() const
{
	return D ;
}

template <typename C>
Tensor<C>* TensorArray<C>::ValArr() const
{
	return V ;
}

template <typename C>
int TensorArray<C>::dimension(int i ) const
{
	return D[i] ;
}

template <typename C>
void TensorArray<C>::display()
{
	std::cout << "Total number of tensors: " << N << std::endl ;
	std::cout << "==============================================" << std::endl ;
	for ( int i = 0; i < N; i ++)
	{
		Tensor<int> ind = arrayPosToTensorInd(i) ;

		std::cout << "tensor array index: ( " ;
		for (int j = 0; j < ind.numel(); j ++)
		{
			std::cout << ind(j) ;
			if ( j < (ind.numel() - 1 ) )
			{
				std::cout << ", " ;
			}
		}
		std::cout << " )" << std::endl ;

		V[i].display() ;
	}
}

template <typename C>
void TensorArray<C>::info()
{
	std::cout << "Total number of tensors: " << N << std::endl ;
	std::cout << "==============================================" << std::endl ;
	for ( int i = 0; i < N; i ++)
	{
		Tensor<int> ind = arrayPosToTensorInd(i) ;

		std::cout << "tensor array index: ( " ;
		for (int j = 0; j < ind.numel(); j ++)
		{
			std::cout << ind(j) ;
			if ( j < (ind.numel() - 1 ) )
			{
				std::cout << ", " ;
			}
		}
		std::cout << " )" << std::endl ;

		V[i].info() ;
		std::cout << "+++++++++++++++++++++++++++++++++++++++++++++" << std::endl ;
	}


}
//---------------------------------------------------------------------------------------------------------

template <typename C>
void TensorArray<C>::save(std::string dirName)
{
	//--------------------------------------------------
	std::string makeDir = "mkdir -p " ;
	makeDir = makeDir + dirName ; // "mkdir -p dirName"
	system( &(makeDir[0]) ) ;
	//---------------------------------
	// delete files in this dir
	std::string rmFile = "rm -rf " + dirName + "/*" ; // "rm -rf dirName/*"
	system( &(rmFile[0]) ) ;
	//++++++++++++++++++++++++++++++++++++
	std::string fileName = dirName + "/info.dat" ;
	std::ofstream outfile(&(fileName[0]), std::ios::binary) ;

	if (!outfile)
	{
		std::cerr<<"save tensor array open error!" << std::endl ;
		exit(0) ;
	}

	outfile.write((char*)&R, sizeof(int)) ;
	outfile.write((char*)&N, sizeof(int)) ;
	if ( R != -1 )
	{
		outfile.write((char*)D, sizeof(int) * R) ;
		for (int i = 0; i < N; i ++)
		{
			fileName = "/t" + num2str(i) + ".dat" ;
			fileName = dirName + fileName ;
			V[i].save(fileName) ;
		}
	}
	outfile.close() ;
	//++++++++++++++++++++++++++++++++++++

}

template <typename C>
void TensorArray<C>::load(std::string dirName)
{
	std::string fileName = dirName + "/info.dat" ;
	std::ifstream infile(&(fileName[0]), std::ios::binary) ;

	if (!infile)
	{
		std::cerr << "load tensor array open error" << std::endl ;
		exit(0) ;
	}
	//***************************************************
	// if original tensor array is not empty, delete D
	if (R > 0)
	{
		delete[] D ;
	}
	infile.read((char*)&R, sizeof(int)) ;
	// //////////////////////////////////////////////////
	// if original tensor array is not empty, delete V
	if (N > 0)
	{
		delete[] V ;
	}
	infile.read((char*)&N, sizeof(int)) ;
	// //////////////////////////////////////////////////
	// if loaded tensor array is not empty, create D
	if (R > 0)
	{
		D = new int[R] ;
		infile.read((char*)D, sizeof(int) * R) ;
	}
	else // if loaded tensor array is empty
	{
		D = NULL ;
	}
	// //////////////////////////////////////////////////
	// if loaded tensor is not empty, create V
	if (N > 0)
	{
		V = new Tensor<C> [N] ;
		for (int i = 0; i < N; i ++)
		{
			fileName = "/t" + num2str(i) + ".dat" ;
			fileName = dirName + fileName ;
			V[i].load(fileName) ;
		}
	}
	else // if loaded tensor is empty
	{
		V = NULL ;
	}
	//***************************************************
	infile.close() ;
}


#endif /* TENSORARRAY_HPP_ */
