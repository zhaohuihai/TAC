/*
 * tensor.cpp
 *
 *  Created on: 2011-5-26
 *      Author: zhaohuihai
 */

#include "main.h"

using namespace std ;
//==================== constructor =================================================
Tensor::Tensor() { }

Tensor::Tensor(valarray<int> &dimension)
{
	dim = dimension ;
	RC = 'C' ; // default: column vector
	// initialize tensor with all elements setting to zero
	createTensor_zero() ;
}
Tensor::Tensor(valarray<int> &dimension, int entryType)
{
	dim = dimension ;
	RC = 'C' ; // default: column vector
	switch (entryType)
	{
	case 0:
		createTensor_zero() ;
		break ;
	case 1:
		createTensor_one() ;
		break ;
	case 2:
		createTensor_randUniform() ;
		break ;
	}
}
Tensor::Tensor(double* a, valarray<int> &dimension)
{
	int n = prod(dimension) ;
	valarray<double> v(a, n) ;
	dim = dimension ;
	RC = 'C' ; // default: column vector
	value = v ;
}
Tensor::Tensor(int n)
{
	valarray<int> dimA(n, 1) ;
	dim = dimA ;
	RC = 'C' ; // default: column vector
	valarray<double> v(n) ;
	value = v ;
}
Tensor::Tensor(int n1, int n2) // create n1 X n2 all zero matrix/vector
{

	if (n2 == 1) // column vector
	{
		valarray<int> dimA(n1, 1) ;
		dim = dimA ;
		RC = 'C' ;
	}
	else if (n1 == 1) // row vector
	{
		valarray<int> dimA(n2, 1) ;
		dim = dimA ;
		RC = 'R' ;
	}
	else
	{
		valarray<int> dimA(2) ;
		dimA[0] = n1 ;
		dimA[1] = n2 ;
		dim = dimA ;
		RC = 'C' ; // default: column vector
	}

	int n = n1 * n2 ;
	valarray<double> v(n) ;
	value = v ;
}
Tensor::Tensor(int n1, int n2, int n3, int n4)
{
	valarray<int> dimA(4) ;
	dimA[0] = n1 ;
	dimA[1] = n2 ;
	dimA[2] = n3 ;
	dimA[3] = n4 ;

	dim = dimA ;

	int n = n1 * n2 * n3 * n4 ;
	valarray<double> v(n) ;
	value = v ;
}

//===================================================================================
// initialize tensor with all elements setting to zero
void Tensor::createTensor_zero()
{
	int n;
	n = prod<int>(dim) ;
	valarray<double> v(n) ;
	value = v ;
}
// initialize tensor with all elements setting to one
void Tensor::createTensor_one()
{
	int n;
	n = prod<int>(dim) ;
	valarray<double> v(1, n) ;
	value = v ;
}
// initialize tensor with all elements setting to uniform distribution random number
void Tensor::createTensor_randUniform()
{
	int n;
	n = prod<int>(dim) ;

	double *r = new double[n]; /* buffer for random numbers */

	unsigned int randSeed = time(0) - 13e8 ;
	//unsigned int randSeed = 3 ;
	VSLStreamStatePtr stream;
	vslNewStream( &stream, VSL_BRNG_MT19937, randSeed );
	/* Generating */
	vdRngUniform( VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, n, r, -0.5, 0.5 );
	/* Deleting the stream */
	vslDeleteStream( &stream );
	valarray<double> v(r, n) ;

	value = v ;

	delete [] r ;
}
//==============================================================
// member function
Tensor& Tensor::operator = (double a)
{
	this->value = a ;
	return *this ;
}

double& Tensor::operator () (int i1)
{
	if (i1 <= 0)
	{
		cout << "Tensor::operator () error: index must be a positive integer." << endl ;
		exit (0) ;
	}
	i1 = i1 - 1 ;
	if (dim.size() == 1)
	{
		int n ;
		n = i1 ;
		return this->value[n] ;
	}
	else
	{
		cerr << "index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
}

double& Tensor::operator () (int i1, int i2)
{
	i1 = i1 - 1 ;
	i2 = i2 - 1 ;
	if (dim.size() == 2)
	{
		int n ;
		n = i1 + i2 * dim[0] ;
		return this->value[n] ;
	}
	else
	{
		cerr << "index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
}
double& Tensor::operator () (int i1, int i2, int i3)
{
	i1 = i1 - 1 ;
	i2 = i2 - 1 ;
	i3 = i3 - 1 ;
	if (dim.size() == 3)
	{
		int n ;
		n = i1 + i2 * dim[0] + i3 * dim[0] * dim[1] ;
		return this->value[n] ;
	}
	else
	{
		cerr << "index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
}
double& Tensor::operator () (int i1, int i2, int i3, int i4)
{
	i1 = i1 - 1 ;
	i2 = i2 - 1 ;
	i3 = i3 - 1 ;
	i4 = i4 - 1 ;
	if (dim.size() == 4)
	{
		int n ;
		n = i1 + i2 * dim[0] + i3 * dim[0] * dim[1] + i4 * dim[0] * dim[1] * dim[2] ;
		return this->value[n] ;
	}
	else
	{
		cerr << "index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
}

double& Tensor::operator () (int i1, int i2, int i3, int i4, int i5)
{
	i1 = i1 - 1 ;
	i2 = i2 - 1 ;
	i3 = i3 - 1 ;
	i4 = i4 - 1 ;
	i5 = i5 - 1 ;
	if (dim.size() == 5)
	{
		int n ;
		n = i1 + i2 * dim[0] + i3 * dim[0] * dim[1] + i4 * dim[0] * dim[1] * dim[2] +
				i5 * dim[0] * dim[1] * dim[2] * dim[3] ;
		return this->value[n] ;
	}
	else
	{
		cerr << "index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
}

double& Tensor::operator () (int i1, int i2, int i3, int i4, int i5, int i6)
{
	i1 = i1 - 1 ;
	i2 = i2 - 1 ;
	i3 = i3 - 1 ;
	i4 = i4 - 1 ;
	i5 = i5 - 1 ;
	i6 = i6 - 1 ;
	if (dim.size() == 6)
	{
		int n ;
		n = i1 + i2 * dim[0] + i3 * dim[0] * dim[1] + i4 * dim[0] * dim[1] * dim[2] +
				i5 * dim[0] * dim[1] * dim[2] * dim[3] +
				i6 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] ;
		return this->value[n] ;
	}
	else
	{
		cerr << "index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
}

double& Tensor::operator () (int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
	i1 = i1 - 1 ;
	i2 = i2 - 1 ;
	i3 = i3 - 1 ;
	i4 = i4 - 1 ;
	i5 = i5 - 1 ;
	i6 = i6 - 1 ;
	i7 = i7 - 1 ;
	if (dim.size() == 7)
	{
		int n ;
		n = i1 + i2 * dim[0] + i3 * dim[0] * dim[1] + i4 * dim[0] * dim[1] * dim[2] +
				i5 * dim[0] * dim[1] * dim[2] * dim[3] +
				i6 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] +
				i7 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] ;
		return this->value[n] ;
	}
	else
	{
		cerr << "index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
}

double& Tensor::operator () (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8)
{
	i1 = i1 - 1 ;
	i2 = i2 - 1 ;
	i3 = i3 - 1 ;
	i4 = i4 - 1 ;
	i5 = i5 - 1 ;
	i6 = i6 - 1 ;
	i7 = i7 - 1 ;
	i8 = i8 - 1 ;
	if (dim.size() == 8)
	{
		int n ;
		n = i1 + i2 * dim[0] + i3 * dim[0] * dim[1] + i4 * dim[0] * dim[1] * dim[2] +
				i5 * dim[0] * dim[1] * dim[2] * dim[3] +
				i6 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] +
				i7 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] +
				i8 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] * dim[6] ;
		return this->value[n] ;
	}
	else
	{
		cerr << "index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
}

double& Tensor::operator () (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11, int i12)
{
	i1 = i1 - 1 ;
	i2 = i2 - 1 ;
	i3 = i3 - 1 ;
	i4 = i4 - 1 ;
	i5 = i5 - 1 ;
	i6 = i6 - 1 ;
	i7 = i7 - 1 ;
	i8 = i8 - 1 ;
	i9 = i9 - 1 ;
	i10 = i10 - 1 ;
	i11 = i11 - 1 ;
	i12 = i12 - 1 ;
	if (dim.size() == 12)
		{
			int n ;
			n = i1 + i2 * dim[0] + i3 * dim[0] * dim[1] + i4 * dim[0] * dim[1] * dim[2] +
					i5 * dim[0] * dim[1] * dim[2] * dim[3] +
					i6 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] +
					i7 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] +
					i8 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] * dim[6] +
					i9 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] * dim[6] * dim[7] +
					i10 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] * dim[6] * dim[7] * dim[8] +
					i11 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] * dim[6] * dim[7] * dim[8] * dim[9] +
					i12 * dim[0] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] * dim[6] * dim[7] * dim[8] * dim[9] * dim[10] ;
			return this->value[n] ;
		}
		else
		{
			cerr << "index and tensor dimension mismatch!" << endl ;
			exit (0) ;
		}
}

double& Tensor::operator () (valarray<int> position)
{
	position = position - 1 ;
	if (dim.size() != position.size())
	{
		cout << "Tensor::operator () error: index and tensor dimension mismatch!" << endl ;
		exit (0) ;
	}
	else
	{
		int i ;
		int index = dim.size() - 1 ;
		int n = position[index] ;
		for (i = index; i > 0 ; i --)
		{
			n = position[i - 1] + n * dim[i - 1] ;
		}
		return this->value[n] ;
	}

}

double& Tensor::operator [] (int n)
{
	if (n <= 0)
	{
		cout << "Tensor::operator [] error: index must be a positive integer." << endl ;
		exit(0) ;
	}
	return this->value[n-1] ;
}

slice Tensor::getSlice(size_t start,  size_t end)
{
	if ((start <= 0) || (end <= 0))
	{
		cout << "Tensor::getSlice error: index must be a positive integer." << endl ;
		exit (0) ;
	}
	start = start - 1 ;
	end = end - 1 ;

	size_t size = end - start + 1 ;
	size_t stride = 1 ;
	return slice(start, size, stride) ;
}

gslice Tensor::getSlice(size_t start1,  size_t end1, size_t start2, size_t end2)
{
	if ((start1 <= 0) || (end1 <= 0) || (start2 <= 0) || (end2 <= 0))
	{
		cout << "Tensor::getSlice error: index must be a positive integer." << endl ;
		exit (0) ;
	}
	start1 = start1 - 1 ;
	end1 = end1 - 1 ;
	start2 = start2 - 1 ;
	end2 = end2 - 1 ;

	if (dim.size() != 2)
	{
		cout << "Tensor::getSlice error: It is not a matrix." << endl ;
		exit(0) ;
	}
	int m = dim[0] ;
	int n = dim[1] ;
	size_t start = start1 + start2 * m ;
	size_t len [] = {(end2 - start2 + 1), (end1 - start1 + 1)} ; // {(No. of columns), (No. of rows)}
	size_t str [] = {m, 1} ;
	valarray<size_t> lengths(len, 2) ;
	valarray<size_t> strides(str, 2) ;
	/*
	valarray<double> v = value[gslice(start, lengths, strides)] ;
	for (int i = 0; i < v.size(); i ++)
	{
		cout << v[i] << "  " ;
	}
	cout << endl ;
	*/
	//gslice slc(start, lengths, strides) ;
	return gslice(start, lengths, strides) ;
}

Tensor Tensor::subTensor(size_t start,  size_t end)
{
	if ((start <= 0) || (end <= 0))
	{
		cout << "Tensor::subTensor error: index must be a positive integer." << endl ;
		exit (0) ;
	}
	start = start - 1 ;
	end = end - 1 ;

	size_t size = end - start + 1 ;
	size_t stride = 1 ;
	Tensor A = Tensor(size) ;
	A.value = value[slice(start, size, stride)] ;

	if (RC == 'R')
	{
		A.trans() ;
	}
	return A ;
}
Tensor Tensor::subTensor(size_t start1,  size_t end1, size_t start2, size_t end2)
{
	// process input
	if ((start1 <= 0) || (end1 <= 0) || (start2 <= 0) || (end2 <= 0))
	{
		cout << "Tensor::subTensor error: index must be a positive integer." << endl ;
		exit (0) ;
	}
	start1 = start1 - 1 ;
	end1 = end1 - 1 ;
	start2 = start2 - 1 ;
	end2 = end2 - 1 ;
	// dimension of the output matrix/vector
	int dim1 = end1 - start1 + 1 ;
	int dim2 = end2 - start2 + 1 ;

	int m = dim[0] ;
	int n = dim[1] ;
	size_t start = start1 + start2 * m ;
	size_t len [] = {dim2, dim1} ;
	size_t str [] = {m, 1} ;
	valarray<size_t> lengths(len, 2) ;
	valarray<size_t> strides(str, 2) ;
	Tensor A = Tensor(dim1, dim2) ;
	A.value = value[gslice(start, lengths, strides)] ;
	return A ;
}
// Rearrange dimensions of N-D array
void Tensor::permute(valarray<int> order)
{
	if (order.min() <= 0)
	{
		cout << "permutation error: permutation indices must be positive integers!" << endl ;
		exit(0) ;
	}

	// convert the starting index to 0
	order = order - 1 ;

	if (dim.size() != order.size())
	{
		cout << "permutation error: array dimension and the extent of ORDER must be the same! " << endl ;
		exit (0) ;
	}
	else
	{
		Tensor A0 = *this ;
		int i ;
		for (i = 0; i < order.size(); i ++)
		{
			dim[i] = A0.dim[ order[i] ] ;
		}
		int n0, n ;
		valarray<int> p0(A0.dim.size());
		valarray<int> p(dim.size()) ;
		for (n0 = 0; n0 < A0.value.size(); n0 ++)
		{
			int index = A0.dim.size() - 1 ;
			// find position of the element (p0) recursively
			// input args: A0.dim, n0, index
			// output args: p0 ;
			findPosition(A0.dim, n0, index, p0) ;

			// find corresponding p
			for (i = 0; i < order.size(); i ++)
			{
				p[i] = p0[ order[i] ] ;
			}

			(*this)(p) = A0(p0) ;
		}
	}
}
void Tensor::permute(int i1, int i2)
{
	valarray<int> order(2) ;
	order[0] = i1 ;
	order[1] = i2 ;

	//this->permute(order) ;
	permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3)
{
	valarray<int> order(3) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4)
{
	valarray<int> order(4) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4, int i5)
{
	valarray<int> order(5) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;
	order[4] = i5 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4, int i5, int i6)
{
	valarray<int> order(6) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;
	order[4] = i5 ;
	order[5] = i6 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
	valarray<int> order(7) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;
	order[4] = i5 ;
	order[5] = i6 ;
	order[6] = i7 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8)
{
	valarray<int> order(8) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;
	order[4] = i5 ;
	order[5] = i6 ;
	order[6] = i7 ;
	order[7] = i8 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9)
{
	valarray<int> order(9) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;
	order[4] = i5 ;
	order[5] = i6 ;
	order[6] = i7 ;
	order[7] = i8 ;
	order[8] = i9 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10)
{
	valarray<int> order(10) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;
	order[4] = i5 ;
	order[5] = i6 ;
	order[6] = i7 ;
	order[7] = i8 ;
	order[8] = i9 ;
	order[9] = i10 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11)
{
	valarray<int> order(11) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;
	order[4] = i5 ;
	order[5] = i6 ;
	order[6] = i7 ;
	order[7] = i8 ;
	order[8] = i9 ;
	order[9] = i10 ;
	order[10] = i11 ;

	this->permute(order) ;
}
void Tensor::permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11, int i12)
{
	valarray<int> order(12) ;
	order[0] = i1 ;
	order[1] = i2 ;
	order[2] = i3 ;
	order[3] = i4 ;
	order[4] = i5 ;
	order[5] = i6 ;
	order[6] = i7 ;
	order[7] = i8 ;
	order[8] = i9 ;
	order[9] = i10 ;
	order[10] = i11 ;
	order[11] = i12 ;

	this->permute(order) ;
}

void Tensor::reshape(valarray<int> &dimension)
{
	if (prod(dim) != prod(dimension))
	{
		cout << "Tensor::reshape error: the number of elements must not change." << endl ;
		exit (0) ;
	}
	else
	{
		dim = dimension ;
	}
}
void Tensor::reshape(int i1)
{
	valarray<int> dimension(1) ;
	dimension[0] = i1 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2)
{
	valarray<int> dimension(2) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3)
{
	valarray<int> dimension(3) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4)
{
	valarray<int> dimension(4) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4, int i5)
{
	valarray<int> dimension(5) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;
	dimension[4] = i5 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4, int i5, int i6)
{
	valarray<int> dimension(6) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;
	dimension[4] = i5 ;
	dimension[5] = i6 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7)
{
	valarray<int> dimension(7) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;
	dimension[4] = i5 ;
	dimension[5] = i6 ;
	dimension[6] = i7 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8)
{
	valarray<int> dimension(8) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;
	dimension[4] = i5 ;
	dimension[5] = i6 ;
	dimension[6] = i7 ;
	dimension[7] = i8 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9)
{
	valarray<int> dimension(9) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;
	dimension[4] = i5 ;
	dimension[5] = i6 ;
	dimension[6] = i7 ;
	dimension[7] = i8 ;
	dimension[8] = i9 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10)
{
	valarray<int> dimension(10) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;
	dimension[4] = i5 ;
	dimension[5] = i6 ;
	dimension[6] = i7 ;
	dimension[7] = i8 ;
	dimension[8] = i9 ;
	dimension[9] = i10 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11)
{
	valarray<int> dimension(11) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;
	dimension[4] = i5 ;
	dimension[5] = i6 ;
	dimension[6] = i7 ;
	dimension[7] = i8 ;
	dimension[8] = i9 ;
	dimension[9] = i10 ;
	dimension[10] = i11 ;

	this->reshape(dimension) ;
}
void Tensor::reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11, int i12)
{
	valarray<int> dimension(12) ;
	dimension[0] = i1 ;
	dimension[1] = i2 ;
	dimension[2] = i3 ;
	dimension[3] = i4 ;
	dimension[4] = i5 ;
	dimension[5] = i6 ;
	dimension[6] = i7 ;
	dimension[7] = i8 ;
	dimension[8] = i9 ;
	dimension[9] = i10 ;
	dimension[10] = i11 ;
	dimension[11] = i12 ;

	this->reshape(dimension) ;
}

// transpose matrix or vector
void Tensor::trans()
{
	// matrix
	if (dim.size() == 2)
	{
		Tensor A0 = *this ;
		(*this).dim[0] = A0.dim[1] ;
		(*this).dim[1] = A0.dim[0] ;
		int i, j ;
		for (i = 1; i <= A0.dim[0]; i ++)
		{
			for (j = 1; j <= A0.dim[1]; j ++)
			{
				(*this)(j, i) = A0(i, j) ;
			}
		}
	}
	// vector
	else if (dim.size() == 1)
	{
		if (RC == 'C')
		{
			RC = 'R' ; // column to row
		}
		else // RC == 'R'
		{
			RC = 'C' ; // row to column
		}
	}
	else
	{
		cout << "Tensor::trans error: transpose on tensor more than 2nd-order is not defined." << endl ;
		exit(0) ;
	}
}

char Tensor::getVecType()
{
	if (dim.size() == 1)
	{
		return RC ;
	}
	else
	{
		cout << "Tensor::getVecType error: It is not a vector." << endl ;
		exit(0) ;
	}
}


// find position of the element (p0) recursively
// input args: dim0, n0, index
// output args: p0 ;
void Tensor::findPosition(valarray<int> &dim0, int n0, int index, valarray<int> &p0)
{
	int ip, i ;
	int step ;
	int a ;
	for (ip = index; ip > 0; ip --)
	{
		step = 1 ;
		for (i = 0; i < ip; i ++)
		{
			step = step * dim0[i] ;
		}
		a = n0 / step ;
		p0[ip] = a ;
		n0 = n0 - a * step ;
	}
	p0[ip] = n0 ;

	p0 = p0 + 1 ;
}

/*
 * normType
 * CHARACTER*1. Specifies the value to be returned by the routine:
 * = 'M' or 'm': val = max(abs(Aij)), largest absolute value of the matrix A.
 * = '1' or 'O' or 'o': val = norm1(A), 1-norm of the matrix A (maximum column sum),
 * = 'I' or 'i': val = normI(A), infinity norm of the matrix A (maximum row sum),
 * = 'F', 'f', 'E' or 'e': val = normF(A), Frobenius norm of the matrix A (square root of sum of squares).
 */
double Tensor::norm(char normType)
{
	double val ;
	if (dim.size() == 2) // matrix norm
	{
		int m = dim[0] ;
		int n = dim[1] ;
		double * a = convertTensor2Array(*this) ;
		double * work = new double [m] ;
		val = dlange(&normType, &m, &n, a, &m, work) ;
		delete [] a ;
		delete [] work ;
	}
	else if (dim.size() == 1)  // vector norm
	{
		int n = dim[0] ;
		int incx = 1 ;
		double * x = convertTensor2Array(*this) ;
		switch (normType)
		{
		case '1':
			val = dasum(&n, x, &incx) ;
			break ;
		case '2':
			val = dnrm2(&n, x, &incx) ;
			break ;
		}
		delete []  x ;
	}
	return val ;
}

double Tensor::sum()
{
	return value.sum() ;
}

double Tensor::max()
{
	return value.max() ;
}

double Tensor::min()
{
	return value.min() ;
}

bool Tensor::isSym()
{
	if ((*this).dim.size() != 2)
	{
		cout << "Tensor::isSym error: this is not a matrix" << endl ;
		exit(0) ;
	}
	int m = (*this).dim[0] ;
	int n = (*this).dim[1] ;
	if (m != n)
	{
		cout << "Tensor::isSym error: matrix is not square" << endl ;
		exit(0) ;
	}
	for (int i = 1; i <= m ; i++)
	{
		for (int j = 1; j <= m ; j++)
		{
			if ((*this)(i, j) != (*this)(j, i) )
			{
				return false ;
			}
		}
	}
	return true ;
}
// =====================================non-member function==========================================
Tensor operator + (Tensor A, Tensor B)
{
	if (isAllDimAgree(A, B))
	{
		A.value = A.value + B.value ;
		return A ;
	}
	else
	{
		cout << "Tensor operator + error: Matrix dimensions must agree." << endl ;
		exit (0) ;
	}
}

Tensor operator - (Tensor A, Tensor B)
{
	if (isAllDimAgree(A, B))
	{
		A.value = A.value - B.value ;
		return A ;
	}
	else
	{
		cout << "Tensor operator - error: Matrix dimensions must agree." << endl ;
		exit (0) ;
	}
}
Tensor operator - (double coef, Tensor A)
{
	A.value = coef - A.value ;
	return A ;
}
Tensor operator * (Tensor A, Tensor B)
{
	int m, k, n ;
	int ka, kb ;
	// A(m,ka)
	if (A.dim.size() == 2)
	{
		m = A.dim[0] ;
		ka = A.dim[1] ;
	}
	else if (A.dim.size() == 1)
	{
		if (A.getVecType() == 'C') // column vector
		{
			m = A.dim[0] ;
			ka = 1 ;
		}
		else // row vector
		{
			m = 1 ;
			ka = A.dim[0] ;
		}
	}
	else
	{
		cout << "Tensor operator * error: inputs must be a matrix or a vector." << endl ;
		exit (0) ;
	}

	// B(kb,n)
	if (B.dim.size() == 2)
	{
		kb = B.dim[0] ;
		n = B.dim[1] ;
	}
	else if (B.dim.size() == 1)
	{
		if (B.getVecType() == 'C') // column vector
		{
			kb = B.dim[0] ;
			n = 1 ;
		}
		else // row vector
		{
			kb = 1 ;
			n = B.dim[0] ;
		}
	}
	else
	{
		cout << "Tensor operator * error: inputs must be a matrix or a vector." << endl ;
		exit (0) ;
	}

	if (ka != kb)
	{
		cout << "Tensor operator * error: inner index dimensions must agree." << endl ;
		exit(0) ;
	}
	// matrix multiplication
	k = ka ;
	// a(m,k)
	double * a = convertTensor2Array(A) ;
	// b(k,n)
	double * b = convertTensor2Array(B) ;
	// ab(m,n)
	double * ab = new double [m * n] ;
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, m, b, k, 0.0, ab, m) ;
	delete [] a ;
	delete [] b ;
	// return
	if (B.dim.size() == 1)
	{
		if (B.getVecType() == 'C')
		{
			valarray<int> dimAB(1) ;
			dimAB[0] = m ;
			Tensor AB = Tensor(ab, dimAB) ; // AB is a column vector
			delete [] ab ;
			return AB ;
		}
	}
	else if (A.dim.size() == 1)
	{
		if (A.getVecType() == 'R')
		{
			valarray<int> dimAB(1) ;
			dimAB[0] = n ;
			Tensor AB = Tensor(ab, dimAB) ;
			delete [] ab ;
			AB.trans() ; // AB is a row vector
			return AB ;
		}
	}
	valarray<int> dimAB(2) ;
	dimAB[0] = m ;
	dimAB[1] = n ;
	Tensor AB = Tensor(ab, dimAB) ;
	delete [] ab ;
	return AB ;

}
Tensor operator * (Tensor A, double coef)
{
	A.value = A.value * coef ;
	return A ;
}
Tensor operator * (double coef, Tensor A)
{
	A.value = A.value * coef ;
	return A ;
}
Tensor operator / (Tensor A, double coef)
{
	A.value = A.value / coef ;
	return A ;
}

Tensor sqrt(Tensor &A)
{
	Tensor sqrtA = A ;
	sqrtA.value = sqrt(A.value) ;
	return sqrtA ;
}

Tensor log10(Tensor &A)
{
	Tensor logA = A ;
	logA.value = log10(A.value) ;
	return logA ;
}

//===============================================================
//destructor
Tensor::~Tensor()
{
	//delete &value ;
	//delete &dim ;
}
