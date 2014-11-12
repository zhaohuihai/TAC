/*
 * test.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: zhaohuihai
 */

#include "main.h"
//#include "test.h"

using namespace std ;
//==================================================================
// matrix multiplication
void test()
{
	int m = 3 ;
	int k = 2 ;
	int n = 3 ;
	double* a = new double[m*k] ;
	int i ;
	for (i = 0; i < (m*k); i ++)
	{
		a[i] = (double)i + 1.0 ;
	}

	//
	int j ;
	for (i = 0; i < m; i ++)
	{
		for (j = 0; j < k; j ++)
		{
			cout << a[i + j * m] << ", " ;
		}
		cout << endl ;
	}
	//

	double* b = new double[k*n] ;
	for (i = 0; i < (k*n); i ++)
	{
		b[i] = (double)i ;
	}
	double * ab1 = new double [m*n] ;
	double * ab2 = new double [m*n] ;
	char trans = 'N' ;
	double alpha = 1 ;
	double beta = 0 ;

	int times = 1 ;
	double time_start = time(0) ;
	for (i = 0; i < times; i ++)
	{
		dgemm(&trans, &trans, &m, &n, &k, &alpha, a, &m, b, &k, &beta, ab1, &m) ;
	}
	cout << endl << "dgemm time = " << (time(0) - time_start ) << " seconds" << endl ;
	time_start = time(0) ;
	for (i = 0; i < times; i ++)
	{
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, m, b, k, 0.0, ab2, m) ;
	}
	cout << endl << "cblas_dgemm time = " << (time(0) - time_start ) << " seconds" << endl ;

	for (i = 0; i < m; i ++)
	{
		for (j = 0; j < n; j ++)
		{
			cout << (ab1[i + j * m] - ab2[i + j * m]) << ", " ;
		}
		cout << endl ;
	}

	for (i = 0; i < m; i ++)
	{
		for (j = 0; j < n; j ++)
		{
			cout << ab1[i + j * m] << ", " ;
		}
		cout << endl ;
	}
	delete [] a ;
	delete [] b ;
	delete [] ab1 ;
	delete [] ab2 ;
}

double test3(double a)
{
	return a ;
}

void test4()
{

	int a = 3 ;
	int b = 4 ;

	int c = min(a, b) ;

	cout << c << endl ;
}


// eig
void testEig()
{
	int n = 3 ;
	// A * Z = Z * W
	Tensor<double> A(n, n) ;
	A.arithmeticSequence(1, 1) ;

	Tensor<double> A1 = A.trans() ;

	A = A + A1 ;
	A = A / 2.0 ;
	A.display() ;

	Tensor<double> U(n, n) ;
	Tensor<double> D(n) ;

	symEig(A, U, D) ;

	A.display() ;
	U.display() ;
	D.display() ;
	//-----------------------------------------
}

void testSVD()
{
	int m = 4 ;
	int n = 4 ;
	Tensor<double> A(m, n) ;
	A.arithmeticSequence(1, 1) ;

	int k = min(m, n) ;

	Tensor<double> U(m, m) ;
	Tensor<double> VT(k, n) ;
	Tensor<double> S(k) ;

	svd(A, U, S, VT) ;

	A.display() ;
	U.display() ;
	VT.display() ;
	S.display() ;



}

void testQR()
{
	int m = 5 ;
	int n = 4 ;
	Tensor<double> A(m, n) ;
	//A.arithmeticSequence(1, 1) ;
	A.randUniform() ;

	int k = min(m, n) ;

	Tensor<double> R = A ;

	qr(A, R) ;

	A.display() ;
	R.display() ;

}

void testNorm()
{

	Tensor<double> T(3,4) ;
	for (int i = 1; i <= T.numel(); i ++)
	{
		T[i] = i ;
		cout << T[i] << endl ;
	}
	T.display() ;
	double val = T.norm('F') ;

	cout << "norm: " << val << endl ;
}
//
void testRand()
{
	int n = 3 ;
	int N = n * n ;
	double* a0 = new double [n*n] ;
	//unsigned int randSeed = time(0) - 13e8 ;
	unsigned int randSeed = 1 ;
	VSLStreamStatePtr stream;
	vslNewStream( &stream, VSL_BRNG_MT19937, randSeed );
	/* Generating */
	vdRngUniform( VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, stream, N, a0, 0.0, 1.0 );
	/* Deleting the stream */
	vslDeleteStream( &stream );

	for (int i = 0; i < N; i ++)
	{
		cout << a0[i] << "\n" ;
	}
}

void testSubTensor()
{
	typedef int dataType ;
	Tensor<dataType> T(9, 8, 7) ;

	for (int i = 1; i <= T.numel(); i ++)
	{
		T[i] = i ;
	}

	Tensor<int> start(3) ;
	Tensor<int> end(3) ;

	start(1) = 2 ;
	end(1) = 4 ;

	start(2) = 4 ;
	end(2) = 6 ;

	start(3) = 3 ;
	end(3) = 4 ;

	Tensor<dataType> Ts = T.subTensor(start, end) ;
//	T = T.subTensor(start, end) ;

//	T.display() ;
//	Ts.display() ;
	cout << Ts.rank() << endl ;
	for (int i = 1; i <= Ts.numel(); i ++)
	{
		cout << T[i] << '\t' ;
		cout << Ts[i] << '\t' ;
		cout << endl ;
	}
}

void testTrans()
{
	typedef double dataType ;
	Tensor<dataType> T(5, 4) ;

	for (int i = 1; i <= T.numel(); i ++)
	{
		T[i] = i ;
	}
	T.display() ;
	Tensor<dataType> T1 = T.trans() ;
	T = T.trans() ;
	T.display() ;

}

void testSqueeze()
{
	typedef double dataType ;
	Tensor<dataType> T(1,1,2) ;

	T.squeeze() ;

	for (int i = 1; i <= T.rank(); i ++)
	{
		cout << "dim " << i << ": " << T.dimension(i) << endl ;
	}
}

void testMin()
{
	typedef double dataType ;
	Tensor<dataType> T(3,2) ;
	T.randUniform() ;

	T.display() ;
	cout << T.min() << endl ;

}

void testPermute()
{
	typedef double dataType ;
	Tensor<dataType> T(3,4) ;
	for (int i = 1; i <= T.numel(); i ++)
	{
		T[i] = i ;
		cout << T[i] << endl ;
	}
	cout << endl ;

	Tensor<int> dim(1) ;
	dim(1) = 24 ;

	T = T.reshape(2, 2, 3) ;

	for (int i = 1; i <= T.rank(); i ++)
	{
		cout << "dim " << i << " = " << T.dimension(i) << endl ;
	}

	for (int i = 1; i <= T.numel(); i ++)
	{
		cout << T[i] << endl ;
	}
//	T.display() ;
	cout << endl ;
}

void testContraction()
{
	int d1 = 3 ;
	int d2 = 2 ;
	int d3 = 3 ;

	Tensor<double> A(2, 3, 2) ;
	A.arithmeticSequence(1, 1) ;

	Tensor<double> B = A ;
	//B.arithmeticSequence(1, 1) ;



	Tensor<double> AB = contractTensors(A, 1, 2, 3, B, 1, 2, 3) ;


//	A.display() ;
//	B.display() ;
	AB.display() ;

}
