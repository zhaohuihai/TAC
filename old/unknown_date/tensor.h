/*
 * tensor.h
 *
 *  Created on: 2011-5-26
 *      Author: zhaohuihai
 */
#ifndef Tensor_H_
#define Tensor_H_

#include <main.h>

using namespace std ;

#define Zero        0
#define One         1
#define RandUniform 2

class Tensor
{

public:
	// constructor
	Tensor() ; // empty tensor
	Tensor(valarray<int> &dimension) ;
	Tensor(valarray<int> &dimension, int entryType) ;
	Tensor(double* a, valarray<int> &dimension) ;
	Tensor(int n) ; // column vector
	Tensor(int n1, int n2) ; // create n1 X n2 all zero matrix
	Tensor(int n1, int n2, int n3, int n4) ;
	// destructor
	~Tensor() ;

	valarray<double> value ;
	//double * value ;
	valarray<int> dim ;

	// overload operators
	Tensor& operator = (double a) ;
	double& operator () (int i1) ;
	double& operator () (int i1, int i2) ;
	double& operator () (int i1, int i2, int i3) ;
	double& operator () (int i1, int i2, int i3, int i4) ;
	double& operator () (int i1, int i2, int i3, int i4, int i5) ;
	double& operator () (int i1, int i2, int i3, int i4, int i5, int i6) ;
	double& operator () (int i1, int i2, int i3, int i4, int i5, int i6, int i7) ;
	double& operator () (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) ;
	//double operator () (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) ;
	//double operator () (int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9) ;
	double& operator () (int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11, int i12) ;
	double& operator () (valarray<int> position) ;
	double& operator [] (int i) ;

	slice getSlice(size_t start,  size_t end) ; // slice of a vector
	gslice getSlice(size_t start1,  size_t end1, size_t start2, size_t end2) ; // slice of a matrix
	Tensor subTensor(size_t start,  size_t end) ; // sub vector
	Tensor subTensor(size_t start1,  size_t end1, size_t start2, size_t end2) ; // sub matrix

	void permute(valarray<int> order) ;
	void permute(int i1, int i2) ;
	void permute(int i1, int i2, int i3) ;
	void permute(int i1, int i2, int i3, int i4) ;
	void permute(int i1, int i2, int i3, int i4, int i5) ;
	void permute(int i1, int i2, int i3, int i4, int i5, int i6) ;
	void permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7) ;
	void permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) ;
	void permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9) ;
	void permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10) ;
	void permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11) ;
	void permute(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11, int i12) ;


	void reshape(valarray<int> &dimension) ;
	void reshape(int i1) ;
	void reshape(int i1, int i2) ;
	void reshape(int i1, int i2, int i3) ;
	void reshape(int i1, int i2, int i3, int i4) ;
	void reshape(int i1, int i2, int i3, int i4, int i5) ;
	void reshape(int i1, int i2, int i3, int i4, int i5, int i6) ;
	void reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7) ;
	void reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8) ;
	void reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9) ;
	void reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10) ;
	void reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11) ;
	void reshape(int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, int i10, int i11, int i12) ;

	void trans() ; // transpose matrix or vector
	char getVecType() ;
	void findPosition(valarray<int> &dim0, int n0, int index, valarray<int> &p0) ;
	double norm(char normType) ;
	double sum() ;
	double max() ;
	double min() ;
	bool isSym() ; // check if it is a symmetric matrix
private:
	char RC ; // identifier of column or row vector
	void createTensor_zero() ;
	void createTensor_one() ;
	void createTensor_randUniform() ;
};

Tensor operator + (Tensor A, Tensor B) ;
Tensor operator - (Tensor A, Tensor B) ;
Tensor operator - (double coef, Tensor A) ;
Tensor operator * (Tensor A, Tensor B) ;
Tensor operator * (Tensor A, double coef) ;
Tensor operator * (double coef, Tensor A) ;
Tensor operator / (Tensor A, double coef) ;

Tensor sqrt(Tensor &A) ;
Tensor log10(Tensor &A) ;


#endif /* TENSOR_H_ */
