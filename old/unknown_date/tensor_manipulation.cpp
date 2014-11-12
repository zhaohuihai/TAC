/*
 * tensorManipulation.cpp
 *
 *  Created on: 2011-5-26
 *      Author: zhaohuihai
 */
#include "main.h"

using namespace std ;

void printVector(Tensor &A)
{
	if (A.dim.size() != 1)
	{
		cout << "printVector error: It is not a vector." << endl ;
		exit(0) ;
	}
	if (A.getVecType() == 'C')
	{
		for (int i = 1; i <= A.dim[0]; i ++)
		{
			cout << A(i) << endl ;
		}
	}
	else // row vector
	{
		for (int i = 1; i <= A.dim[0]; i ++)
		{
			cout << A(i) << ",\t" ;
		}
		cout << endl ;
	}
}

void printMatrix(Tensor &A)
{
	if (A.dim.size() != 2)
	{
		cout << "printMatrix error: It is not a matrix" << endl ;
		exit (0) ;
	}
	else
	{
		for (int i = 1; i <= A.dim[0]; i ++)
		{
			for (int j = 1; j <= A.dim[1]; j ++)
			{
				cout << A(i,j) << "\t" ;
			}
			cout << endl ;
		}
	}
}
bool isAllDimAgree(Tensor &A, Tensor &B)
{
	valarray<bool> comp = (A.dim == B.dim) ;
	if (comp.min() == true)
		return true ;
	else
		return false ;
}

double norm(valarray<double> &V)
{
	int n = V.size() ;
	int incx = 1 ;
	double * x = new double [n] ;
	int i ;
	for (i = 0; i < n; i ++)
	{
		x[i] = V[i] ;
	}
	double val = dnrm2(&n, x, &incx) ;
	delete []  x ;
	return val ;
}

Tensor contractTensors(Tensor &A, int a1, Tensor &B, int b1)
{
	valarray<int> indA_right(1) ;
	indA_right[0] = a1 ;

	valarray<int> indB_left(1) ;
	indB_left[0] = b1 ;

	Tensor AB = contractTensors(A, indA_right, B, indB_left) ;
	//printMatrix(AB) ;
	//exit(0) ;
	return AB ;
}
Tensor contractTensors(Tensor &A, int a1, int a2, Tensor &B, int b1, int b2)
{
	valarray<int> indA_right(2) ;
	indA_right[0] = a1 ;
	indA_right[1] = a2 ;

	valarray<int> indB_left(2) ;
	indB_left[0] = b1 ;
	indB_left[1] = b2 ;

	Tensor AB = contractTensors(A, indA_right, B, indB_left) ;

	return AB ;
}
Tensor contractTensors(Tensor &A, int a1, int a2, int a3, Tensor &B, int b1, int b2, int b3)
{
	valarray<int> indA_right(3) ;
	indA_right[0] = a1 ;
	indA_right[1] = a2 ;
	indA_right[2] = a3 ;

	valarray<int> indB_left(3) ;
	indB_left[0] = b1 ;
	indB_left[1] = b2 ;
	indB_left[2] = b3 ;

	Tensor AB = contractTensors(A, indA_right, B, indB_left) ;

	return AB ;
}
Tensor contractTensors(Tensor &A, int a1, int a2, int a3, int a4, Tensor &B, int b1, int b2, int b3, int b4)
{
	valarray<int> indA_right(4) ;
	indA_right[0] = a1 ;
	indA_right[1] = a2 ;
	indA_right[2] = a3 ;
	indA_right[3] = a4 ;

	valarray<int> indB_left(4) ;
	indB_left[0] = b1 ;
	indB_left[1] = b2 ;
	indB_left[2] = b3 ;
	indB_left[3] = b4 ;

	Tensor AB = contractTensors(A, indA_right, B, indB_left) ;

	return AB ;
}

Tensor contractTensors(Tensor A, valarray<int> indA_right, Tensor B, valarray<int> indB_left)
{
	// convert the starting index to 0
	indA_right = indA_right - 1 ;
	indB_left = indB_left - 1 ;
	// original dim of A, B
	valarray<int> dimA = A.dim ;
	valarray<int> dimB = B.dim ;
	// A(indA_left, indA_right)
	valarray<int> indA_left = findOuterIndex(dimA.size(), indA_right) ;
	//cout << indA_left.size() << endl ;
	//cout << indA_left[0] << endl ;
	//exit(0) ;
	// B(indB_left, indB_right)
	valarray<int> indB_right = findOuterIndex(dimB.size(), indB_left) ;
	// A is both input and output arg
	convertTensor2Matrix(A, indA_left, indA_right) ; // including index permutation

	convertTensor2Matrix(B, indB_left, indB_right) ; // including index permutation
	// AB = A * B
	Tensor AB = computeMatrixProduct(A, B) ;

	convertMatrix2Tensor(AB, dimA, indA_left, dimB, indB_right) ;

	return AB ;
}

valarray<int> findOuterIndex(int size, valarray<int> &inner_index)
{
	valarray<int> outer_index(size - inner_index.size()) ;
	int i, j ;
	bool existence = false ;
	int k = 0 ;
	for (i = 0; i < size; i ++)
	{
		for (j = 0; j < inner_index.size(); j ++)
		{
			if (i == inner_index[j])
			{
				//cout << "i = " << i << endl ;
				existence = true ;
				break ;
			}
		}
		if (existence == false)
		{
			outer_index[k] = i ;

			k ++ ;
		}
		else
		{
			existence = false ;
		}
	}
	return outer_index ;
}
// row and column vectors are all regarded as matrices.
void convertTensor2Matrix(Tensor &A, valarray<int> &indA_left, valarray<int> &indA_right)
{
	valarray<int> dimA(2) ; // dimA=[0,0]
	dimA[0] = prod(A.dim, indA_left) ;
	dimA[1] = prod(A.dim, indA_right) ;
	// new index order of A
	valarray<int> orderA(A.dim.size()) ;
	int i ;
	int iA = 0 ;
	for (i = 0; i < indA_left.size(); i ++)
	{
		orderA[iA] = indA_left[i] ;
		iA ++ ;
	}
	for (i = 0; i < indA_right.size(); i ++)
	{
		orderA[iA] = indA_right[i] ;
		iA ++ ;
	}

	//cout << orderA[0] << "\t" << orderA[1] << endl ;
	//exit(0) ;
	orderA = orderA + 1 ;
	A.permute(orderA) ;
	// reshape
	A.dim = dimA ;
}
Tensor computeMatrixProduct(Tensor &A, Tensor &B)
{
	if (A.dim[1] != B.dim[0])
	{
		cout << "Inner matrix dimensions must agree." << endl ;
		cout << "Error at: computeMatrixProduct" << endl ;
		exit(0) ;
	}
	// AB(m,n) = A(m,k)*B(k,n)
	int m = A.dim[0] ;
	int k = A.dim[1] ;
	int n = B.dim[1] ;
	double * a = convertTensor2Array(A) ;
	double * b = convertTensor2Array(B) ;
	double * ab = new double [A.dim[0] * B.dim[1]] ;
	//cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
	//const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B,
	//const int ldb, const double beta, double *C, const int ldc)
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, m, b, k, 0.0, ab, m) ;
	delete [] a ;
	delete [] b ;
	valarray<int> dimAB(2) ;
	dimAB[0] = A.dim[0] ;
	dimAB[1] = B.dim[1] ;
	Tensor AB = Tensor(ab, dimAB) ;
	delete [] ab ;
	return AB ;
}
void convertMatrix2Tensor(Tensor &AB, valarray<int> &dimA, valarray<int> &indA_left, valarray<int> &dimB, valarray<int> &indB_right)
{
	valarray<int> dimAB(indA_left.size() + indB_right.size()) ;
	int i ;
	int iAB = 0 ;
	for (i = 0; i < indA_left.size(); i ++)
	{
		dimAB[iAB] = dimA[indA_left[i]] ;
		iAB ++ ;
	}
	for (i = 0; i < indB_right.size(); i ++)
	{
		dimAB[iAB] = dimB[indB_right[i]] ;
		iAB ++ ;
	}
	// reshape
	AB.dim = dimAB ;
}
double * convertTensor2Array(Tensor &A)
{
	int n = prod(A.dim) ;
	double * a = new double [n] ;
	int i ;
	for (i = 0; i < n; i ++)
	{
		a[i] = A.value[i] ;
	}
	return a ;
}
/*
 *  Compute all the eigenvalues, and the eigenvectors of a real generalized symmetric-definite eigenproblem
 *  A * U = B * U * D
 *   A and B are assumed to be symmetric and B is also positive definite.
 */
Tensor symGenEig(Tensor &A, Tensor &B, Tensor &U)
{
	// matrix dimension
	int n = A.dim[0] ;


	double * a = convertTensor2Array(A) ;
	double * b = convertTensor2Array(B) ;
	double * d = new double [n] ;

	int itype = 1 ; // the problem type is A*x = lambda*B*x
	char jobz = 'V' ;
	char uplo = 'U' ;
	LAPACKE_dsygv(LAPACK_COL_MAJOR, itype, jobz, uplo, n, a, n, b, n, d) ;

	valarray<double> Uval(a, n*n) ;
	U.value = Uval ;
	delete [] a ;
	delete [] b ;

	valarray<int> Ddim(n,1) ;
	Tensor D = Tensor(d, Ddim) ;
	delete [] d ;

	return D ;
}
/*
 *  Computes selected eigenvalues and eigenvectors of a real generalized symmetric definite eigenproblem.
 *  A * U = B * U * D
 *   A and B are assumed to be symmetric and B is also positive definite.
 */
Tensor symGenEig(Tensor &A, Tensor &B, Tensor &U, int k)
{
	// matrix dimension
	int n = A.dim[0] ;

	double * a = convertTensor2Array(A) ;
	double * b = convertTensor2Array(B) ;
	double * w = new double [n] ; // The first m elements of w contain the selected eigenvalues in ascending order.
	double * z = new double [n*k] ; // the first m columns of z contain the orthonormal eigenvectors
	int * ifail = new int [n] ; //

	int itype = 1 ; // the problem type is A*x = lambda*B*x
	char jobz = 'V' ; // compute eigenvalues and eigenvectors.
	char range = 'I' ; // computes eigenvalues with indices il to iu.
	char uplo = 'U' ; // arrays a and b store the upper triangles of A and B
	double abstol = 0 ; // If abstol is less than or equal to zero, then ε*||T||1 is used as tolerance
	int m ; // The total number of eigenvalues found

	/*
	 * lapack_int LAPACKE_<?>sygvx( int matrix_order, lapack_int itype, char jobz, char range, char uplo, lapack_int n, <datatype>* a,
	 * lapack_int lda, <datatype>* b, lapack_int ldb, <datatype> vl, <datatype> vu, lapack_int il, lapack_int iu, <datatype> abstol,
	 * lapack_int* m, <datatype>* w, <datatype>* z, lapack_int ldz, lapack_int* ifail );
	 */
	int info = LAPACKE_dsygvx(LAPACK_COL_MAJOR, itype, jobz, range, uplo, n, a, n, b, n, 0, 1, 1, k, abstol, &m, w, z, n, ifail) ;
	if (info != 0)
	{
		cout << "info = " << info << endl ;
	}

	delete [] a ;
	delete [] b ;
	valarray<double> Uval(z, n * k) ;
	U.value = Uval ;
	delete [] z ;

	valarray<int> Ddim(k, 1) ;
	Tensor D = Tensor(w, Ddim) ;
	delete [] w ;

	delete [] ifail ;

	return D ;
}

/*
 * Computes eigenvalues and eigenvectors of a real symmetric matrix using the Relatively Robust Representations.
 * A * U = U * D
 */
Tensor symEig(Tensor &A, Tensor &U)
{
	// matrix dimension
	int n = A.dim[0] ;

	double * a = convertTensor2Array(A) ;
	double * w = new double [n] ;
	double * z = new double [n*n] ;

	int * isuppz = new int [2 * n] ;

	int m ;
	/*
	 * lapack_int LAPACKE_dsyevr( int matrix_order, char jobz, char range, char uplo, lapack_int n, <datatype>* a,
	 * lapack_int lda, <datatype> vl, <datatype> vu, lapack_int il, lapack_int iu, <datatype> abstol, lapack_int* m,
	 * <datatype>* w, <datatype>* z, lapack_int ldz, lapack_int* isuppz );
	 */
	int info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'A', 'U', n, a, n, 0, 0, 0, 0, 0, &m, w, z, n, isuppz) ;
	if (info != 0)
	{
		cout << "info = " << info << endl ;
	}
	delete [] a ;

	valarray<int> Ddim(n, 1) ;
	Tensor D = Tensor(w, Ddim) ;
	delete [] w ;

	valarray<double> Uval(z, n * n) ;
	U.value = Uval ;
	delete [] z ;
	delete [] isuppz ;

	return D ;
}

// Upper triangular part of matrix
Tensor triu(Tensor A)
{
	if (A.dim.size() != 2)
	{
		cout << "triu error: First input must be a matrix" << endl ;
		exit (0) ;
	}
	int M = A.dim[0] ;
	int N = A.dim[1] ;
	for (int i = 1; i <= M; i ++)
	{
		for (int j = 1; j <= N; j ++)
		{
			if ( i > j)
			{
				A(i, j) = 0 ;
			}
		}
	}
	return A ;
}
Tensor triu(Tensor A, int k)
{
	if (A.dim.size() != 2)
	{
		cout << "triu error: First input must be a matrix" << endl ;
		exit (0) ;
	}
	int M = A.dim[0] ;
	int N = A.dim[1] ;
	for (int i = 1; i <= M; i ++)
	{
		for (int j = 1; j <= N; j ++)
		{
			if ( (i + k) > j)
			{
				A(i, j) = 0 ;
			}
		}
	}
	return A ;
}

Tensor trans(Tensor A)
{
	A.trans() ;
	return A ;
}
