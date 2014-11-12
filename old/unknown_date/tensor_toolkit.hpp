#ifndef _MY_TENSOR_CALCULATION_TOOLKIT_HPP_
#define _MY_TENSOR_CALCULATION_TOOLKIT_HPP_
/* Tensor Toolkit
 * Copyright (C) 2013 Kenji Harada
 * Date: Feb 4, 2013
 */

/**
   @file tensor_toolkit.hpp
   @brief Tensor class and functions for tensor calculations.
   @author Kenji Harada
 */

/**
   @mainpage Toolkit for tensor calculations

   This is a C++ toolkit for tensor calculations. The class Tensor and
   useful functions with class Tensor are defined.  Functions for a
   tensor contraction and a higher-order singular decomposition(HOSVD)
   are defined. All class and functions are defined in the namespace
   TensorToolkit. The detail description are in the page
   of TensorToolkit.

   Feb 4, 2013\n
   Kenji Harada

   @note Class Tensor does not depend another libraries as BLAS and
   LAPACK. The function of tensor contraction can use BLAS. The
   function of HOSVD uses LAPACK. I define two macros as HAVE_LAPACK
   and HAVE_BLAS. If you have LAPACK, please define HAVE_LAPACK that
   automatically defines HAVE_BLAS. If and only if you have BLAS
   without LAPACK, please define HAVE_BLAS.
*/
#include <sstream>
#include <string>
#include <vector>
#include <cstdarg>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <complex>
#include <utility>

/* Header for BLAS and LAPACK */
#ifdef HAVE_LAPACK
#define HAVE_BLAS
#define USE_DIRECTLY_LAPACK
#endif

#define DOUBLE_COMPLEX std::complex < double >

#ifdef HAVE_BLAS
extern "C" {
// BLAS 1
int zswap_(int *N, DOUBLE_COMPLEX *ZX, int *INCX, DOUBLE_COMPLEX *ZY, int *INCY);
int dswap_(int *N, double *ZX, int *INCX, double *ZY, int *INCY);
double ddot_(int *N, double *X, int *INCX, double *Y, int *INCY);
DOUBLE_COMPLEX zdotc_(int *N, DOUBLE_COMPLEX *X, int *INCX, DOUBLE_COMPLEX *Y, int *INCY);
double dnrm2_(int *N, double *X, int *INCX);
double dznrm2_(int *N, DOUBLE_COMPLEX *X, int *INCX);
#define ZSWAP zswap_
#define DSWAP dswap_
  #define DDOT ddot_
  #define ZDOTC zdotc_
  #define DNRM2 dnrm2_
  #define DZNRM2 dznrm2_
// BLAS 3
/// Matrix - Matrix multiplication (complex)
int zgemm_(char* TRANSA, char* TRANSB, int *M, int *N, int *K,
           DOUBLE_COMPLEX *ALPHA,
           DOUBLE_COMPLEX *A, int *LDA,
           DOUBLE_COMPLEX *B, int *LDB,
           DOUBLE_COMPLEX *BETA,
           DOUBLE_COMPLEX *C, int *LDC);
/// Matrix - Matrix multiplication (real)
int dgemm_(char* TRANSA, char* TRANSB, int *M, int *N, int *K,
           double *ALPHA,
           double *A, int *LDA,
           double *B, int *LDB,
           double *BETA,
           double *C, int *LDC);
/// Symmetic operation of complex matrix
int zherk_(char* UPLO, char* TRANS, int *N, int *K, double *ALPHA,
           DOUBLE_COMPLEX *A, int *LDA, double *BETA, DOUBLE_COMPLEX *C, int *LDC);
/// Symmetic operation of real matrix
int dsyrk_(char* UPLO, char* TRANS, int *N, int *K, double *ALPHA,
           double *A, int *LDA, double *BETA, double *C, int *LDC);
}
#define ZGEMM zgemm_
#define DGEMM dgemm_
#define ZHERK zherk_
#define DSYRK dsyrk_
#endif

#ifdef HAVE_LAPACK
extern "C" {
// LAPACK
/// Singular value decomposition of complex matrix
int zgesvd_(char *JOBU, char *JOBVT, int *M, int *N, DOUBLE_COMPLEX *A, int *LDA, double *S,
            DOUBLE_COMPLEX *U, int *LDU, DOUBLE_COMPLEX *VT, int *LDVT,
            DOUBLE_COMPLEX *WORK, int *LWORK, double *RWORK, int *INFO);
/// Singular value decomposition of real matrix
int dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA, double *S, double *U,
            int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO);
/// Diagonalization of hermitian matrix
int zheevd_(char *JOBZ, char *UPLO, int *N, DOUBLE_COMPLEX *A,
            int *LDA, double *W, DOUBLE_COMPLEX *WORK, int *LWORK,
            double *RWORK, int *LRWORK, int *IWORK, int *LIWORK, int *INFO);
/// Diagonalization of real symmetric matrix
int dsyevd_(char *JOBZ, char *UPLO, int *N, double *A,
            int *LDA, double *W, double *WORK, int *LWORK,
            int *IWORK, int *LIWORK, int *INFO);
/// Diagonalization of hermitial matrix for selected eigenvalues
int zheevx_(char *JOBZ, char *RANGE, char *UPLO, int *N, DOUBLE_COMPLEX *A, int *LDA,
            double *VL, double *VU, int *IL, int *IU, double *ABSTOL, int *M, double *W,
            DOUBLE_COMPLEX *Z, int *LDZ, DOUBLE_COMPLEX *WORK, int *LWORK, double *RWORK, int *IWORK,
            int *IFAIL, int *INFO);
/// Diagonalization of real symmetric matrix for selected eigenvalues
int dsyevx_(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A, int *LDA,
            double *VL, double *VU, int *IL, int *IU, double *ABSTOL, int *M, double *W,
            double *Z, int *LDZ, double *WORK, int *LWORK, int *IWORK,
            int *IFAIL, int *INFO);
}
#define DGESVD dgesvd_
#define DSYEVD dsyevd_
#define DSYEVX dsyevx_
#define ZGESVD zgesvd_
#define ZHEEVD zheevd_
#define ZHEEVX zheevx_
#endif

/* Tensor Toolkit */
/**
   @namespace TensorToolkit
   @brief Toolkit for tensor calculations
   @note Class Tensor does not depend another libraries as BLAS and
   LAPACK.
*/
namespace TensorToolkit {
  // Type
  typedef std::complex<double> double_complex;

  /**
     @class Tensor
     @brief Class for tensor
   */
  template <typename C>
  class Tensor {
private:
    /// Rank of tensor
    int R;
    int RMAX;
    /// Array of dimension of indexes
    int* D;
    /// Total number of elements
    int N;
    int NMAX;
    /// Array of values of elements
    C* V;

public:
    Tensor() : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
    }
    Tensor(int r, int d) : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
      initialize(r, d);
    }
    Tensor(int r, const int* d) : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
      initialize(r, d);
    }
    Tensor(const std::vector<int> &d) : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
      initialize(d);
    }
    Tensor(int r, int d, C x) : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
      initialize(r, d, x);
    }
    Tensor(int r, const int* d, C x) : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
      initialize(r, d, x);
    }
    Tensor(const std::vector<int> &d, C x) : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
      initialize(d, x);
    }
    Tensor(const Tensor<C>& T) : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
      copy(T);
    }
    Tensor(const Tensor<C>& T, C x) : R(-1), RMAX(-1), D(NULL), N(0), NMAX(0), V(NULL){
      resize(T);
      for (int i = 0; i < N; ++i) V[i] = x;
    }
    ~Tensor(){
      clear();
    }

    C element(const int i) const {
      return V[i];
    }

    C at(const int i1, ...) const {
      if (R == 0) return V[0];
      va_list ap;
      va_start(ap, i1);
      int ip = i1;
      int base = D[0];
      for (int i = 1; i < R; ++i) {
        int x = va_arg(ap, int);
        ip += x * base;
        base *= D[i];
      }
      va_end(ap);
      return V[ip];
    }

    C at(const int* d) const {
      int ix = d[R - 1];
      for (int i = R - 2; i >= 0; i--) {
        ix *= D[i];
        ix += d[i];
      }
      return V[ix];
    }

    C at(const std::vector<int> &d) const {
      int ix = d[R - 1];
      for (int i = R - 2; i >= 0; i--) {
        ix *= D[i];
        ix += d[i];
      }
      return V[ix];
    }

    C & operator [](int i) {
      return V[i];
    }

    Tensor & operator =(const Tensor<C>& T){
      copy(T);
      return *this;
    }

    /**
       @brief Access operator of an element

       Fortran Order : (i1,i2, ...) -> i1 + i2 * D[0] + i3*D[0]*D[1] + ...
    */
    C & operator ()(const int i1, ...) {
      if (R == 0) return V[0];
      va_list ap;
      va_start(ap, i1);
      int ip = i1;
      int base = D[0];
      for (int i = 1; i < R; ++i) {
        int x = va_arg(ap, int);
        ip += x * base;
        base *= D[i];
      }
      va_end(ap);
      return V[ip];
    }

    C & operator ()(const int* ind) {
      if (R == 0) return V[0];
      int ix = ind[R - 1];
      for (int i = R - 2; i >= 0; i--) {
        ix *= D[i];
        ix += ind[i];
      }
      return V[ix];
    }

    C & operator ()(const std::vector<int> &ind) {
      if (R == 0) return V[0];
      int ix = ind[R - 1];
      for (int i = R - 2; i >= 0; i--) {
        ix *= D[i];
        ix += ind[i];
      }
      return V[ix];
    }

    void initialize(int r, int d){
      int size = 1;
      for (int i = 0; i < r; ++i) size *= d;
      if (r > 0 && RMAX < r) {
        if (RMAX > 0) delete[] D;
        D = new int[r];
        RMAX = r;
      }
      if (NMAX < size) {
        if (NMAX > 0) delete[] V;
        V = new C[size];
        NMAX = size;
      }
      R = r;
      for (int i = 0; i < R; ++i) D[i] = d;
      N = size;
      for (int i = 0; i < N; ++i) V[i] = 0;
    }

    void initialize(int r, const int *d){
      int size = 1;
      for (int i = 0; i < r; ++i) size *= d[i];
      if (r > 0 && RMAX < r) {
        if (RMAX > 0) delete[] D;
        D = new int[r];
        RMAX = r;
      }
      if (NMAX < size) {
        if (NMAX > 0) delete[] V;
        V = new C[size];
        NMAX = size;
      }
      R = r;
      for (int i = 0; i < R; ++i) D[i] = d[i];
      N = size;
      for (int i = 0; i < N; ++i) V[i] = 0;
    }

    void initialize(const std::vector<int> &d){
      int r = d.size();
      int size = 1;
      for (int i = 0; i < r; ++i) size *= d[i];
      if (r > 0 && RMAX < r) {
        if (RMAX > 0) delete[] D;
        D = new int[r];
        RMAX = r;
      }
      if (NMAX < size) {
        if (NMAX > 0) delete[] V;
        V = new C[size];
        NMAX = size;
      }
      R = r;
      for (int i = 0; i < R; ++i) D[i] = d[i];
      N = size;
      for (int i = 0; i < N; ++i) V[i] = 0;
    }

    void initialize(int r, int d, C x){
      int size = 1;
      for (int i = 0; i < r; ++i) size *= d;
      if (r > 0 && RMAX < r) {
        if (RMAX > 0) delete[] D;
        D = new int[r];
        RMAX = r;
      }
      if (NMAX < size) {
        if (NMAX > 0) delete[] V;
        V = new C[size];
        NMAX = size;
      }
      R = r;
      for (int i = 0; i < R; ++i) D[i] = d;
      N = size;
      for (int i = 0; i < N; ++i) V[i] = x;
    }

    void initialize(int r, const int *d, C x){
      int size = 1;
      for (int i = 0; i < r; ++i) size *= d[i];
      if (r > 0 && RMAX < r) {
        if (RMAX > 0) delete[] D;
        D = new int[r];
        RMAX = r;
      }
      if (NMAX < size) {
        if (NMAX > 0) delete[] V;
        V = new C[size];
        NMAX = size;
      }
      R = r;
      for (int i = 0; i < R; ++i) D[i] = d[i];
      N = size;
      for (int i = 0; i < N; ++i) V[i] = x;
    }

    void initialize(const std::vector<int> &d, C x){
      int r = d.size();
      int size = 1;
      for (int i = 0; i < r; ++i) size *= d[i];
      if (r > 0 && RMAX < r) {
        if (RMAX > 0) delete[] D;
        D = new int[r];
        RMAX = r;
      }
      if (NMAX < size) {
        if (NMAX > 0) delete[] V;
        V = new C[size];
        NMAX = size;
      }
      R = r;
      for (int i = 0; i < R; ++i) D[i] = d[i];
      N = size;
      for (int i = 0; i < N; ++i) V[i] = x;
    }

    void set(C x){
      for (int i = 0; i < N; ++i) V[i] = x;
    }

    void clear(){
      if (NMAX > 0) delete[] V;
      if (RMAX > 0) delete[] D;
      R = -1;
      RMAX = -1;
      D = NULL;
      N = 0;
      NMAX = 0;
      V = NULL;
    }

    void copy(const Tensor<C>& T){
      if (T.rank() > 0 && RMAX < T.rank()) {
        if (RMAX > 0) delete[] D;
        D = new int[T.rank()];
        RMAX = T.rank();
      }
      if (NMAX < T.size()) {
        if (NMAX > 0) delete[] V;
        V = new C[T.size()];
        NMAX = T.size();
      }
      R = T.rank();
      for (int i = 0; i < R; ++i) D[i] = T.dimension(i);
      N = T.size();
      for (int i = 0; i < N; ++i) V[i] = T.element(i);
    }

    /**
       @brief
    */
    void reindex(int r, const int *d){
      int v = 1;
      for (int i = 0; i < r; ++i) v *= d[i];
      assert(v == N);
      resize(r, d);
    }

    void reindex(const std::vector<int> &d){
      int v = 1;
      for (std::vector<int>::const_iterator it = d.begin(); it != d.end(); ++it) v *= (*it);
      assert(v == N);
      resize(d);
    }

    void resize(int r, const int *d){
      int size = 1;
      for (int i = 0; i < r; ++i) size *= d[i];
      if (r > 0 && RMAX < r) {
        if (RMAX > 0) delete[] D;
        D = new int[r];
        RMAX = r;
      }
      if (NMAX < size) {
        if (NMAX > 0) delete[] V;
        V = new C[size];
        NMAX = size;
      }
      R = r;
      for (int i = 0; i < R; ++i) D[i] = d[i];
      N = size;
    }

    void resize(const std::vector<int> &d){
      int r = d.size();
      int size = 1;
      for (int i = 0; i < r; ++i) size *= d[i];
      if (r > 0 && RMAX < r) {
        if (RMAX > 0) delete[] D;
        D = new int[r];
        RMAX = r;
      }
      if (NMAX < size) {
        if (NMAX > 0) delete[] V;
        V = new C[size];
        NMAX = size;
      }
      R = r;
      for (int i = 0; i < R; ++i) D[i] = d[i];
      N = size;
    }

    void resize(const Tensor<C>& T){
      if (T.rank() > 0 && RMAX < T.rank()) {
        if (RMAX > 0) delete[] D;
        D = new int[T.rank()];
        RMAX = T.rank();
      }
      if (NMAX < T.size()) {
        if (NMAX > 0) delete[] V;
        V = new C[T.size()];
        NMAX = T.size();
      }
      R = T.rank();
      for (int i = 0; i < R; ++i) D[i] = T.dimension(i);
      N = T.size();
    }

    void set_size(int r, ...){
      va_list ap;
      std::vector<int> dim(r);
      va_start(ap, r);
      for (int i = 0; i < r; ++i) {
        int x = va_arg(ap, int);
        dim[i] = x;
      }
      va_end(ap);
      initialize(r, &(dim[0]));
    }

    void add(const Tensor<C>& T){
      assert(N == T.size());
      for (int i = 0; i < N; ++i) V[i] += T.element(i);
    }

    int rank() const {
      return R;
    }
    int size() const {
      return N;
    }
    int dimension(int i) const {
      return D[i];
    }

  };
  // New type
  typedef Tensor< double_complex > CTensor;
  typedef Tensor< double > RTensor;

  /**
     @class Forall
     @brief Enumeration of all elements

     Prepare two counters (index1 and index2) as the follows:
     @code
     for(x_1=0;x_1<dim[0];++x_1)
       ...
         for(x_N=0;x_N<dim[N-1];++x_N){
           index1 = x_1*base1[0] + ... x_N * base1[N-1];
           index2 = x_1*base2[0] + ... x_N * base2[N-1];
         }
       ...
     }
     @endcode
  */
  class Forall {
private:
    std::vector<int> base1;
    std::vector<int> base2;
    std::vector<int> dim;
    std::vector<int> sum1;
    std::vector<int> sum2;
    std::vector<int> x;
    int N;
public:
    int index1, index2;
public:
    Forall(){
      clear();
    }

    /**
       @brief Reset
     */
    void clear(){
      N = 0;
      base1.clear();
      base2.clear();
      dim.clear();
      sum1.clear();
      sum2.clear();
      x.clear();
      base1.push_back(0);
      base2.push_back(0);
      dim.push_back(0);
      sum1.push_back(0);
      sum2.push_back(0);
      x.push_back(1);
      index1 = 0;
      index2 = 0;
    }

    /**
       @brief Load setup information

       @param[in] num Number of indexes
       @param[in] xbase1 Array of bases of indexes of No. 1
       @param[in] xbase2 Array of bases of indexes of No. 2
       @param[in] xdim Dimension of indexes
     */
    void set_up(int num, const int *xbase1, const int *xbase2, const int* xdim){
      base1.resize(num);
      base2.resize(num);
      dim.resize(num);
      sum1.resize(num, 0);
      sum2.resize(num, 0);
      x.resize(num);
      N = num;
      for (int i = 0; i < N; ++i) {
        base1[i] = xbase1[i];
        base2[i] = xbase2[i];
        dim[i] = xdim[i];
        x[i] = xdim[i] - 1;
      }
    }

    /**
       @brief Load setup information

       @param[in] xbase1 Array of bases of indexes of No. 1
       @param[in] xbase2 Array of bases of indexes of No. 2
       @param[in] xdim Dimension of indexes
     */
    void set_up(const std::vector<int> &xbase1, const std::vector<int> &xbase2, const std::vector<int> &xdim){
      int num = xbase1.size();
      base1 = xbase1;
      base2 = xbase2;
      dim = xdim;
      sum1.resize(num, 0);
      sum2.resize(num, 0);
      x.resize(num);
      N = num;
      for (int i = 0; i < N; ++i)
        x[i] = dim[i] - 1;
    }

    /**
       @brief Set information of an index

       @param[in] xbase1 Base of an index of No. 1
       @param[in] xbase2 Base of an index of No. 2
       @param[in] xdim Dimension of an index
     */
    void push_back(int xbase1, int xbase2, int xdim){
      base1.push_back(xbase1);
      base2.push_back(xbase2);
      dim.push_back(xdim);
      sum1.push_back(0);
      sum2.push_back(0);
      x.push_back(xdim - 1);
      N++;
    }

    /**
       @brief Start an enumeration

       Reset counters (index1 and index2)
     */
    void start(){
      x[0] = 1;
      sum1[0] = 0;
      sum2[0] = 0;
      for (int i = 1; i <= N; ++i) {
        x[i] = dim[i] - 1;
        sum1[i] = 0;
        sum2[i] = 0;
      }
      index1 = sum1[N];
      index2 = sum2[N];
    }

    /**
       @brief Check an end of enumeration

       @return When counter goes to an end, return false.
     */
    bool check(){
      return x[0] != 0;
    }

    /**
       @brief Update counters (index1 and index2)
    */
    void next(){
      int il = N;
      while (x[il] == 0)
        il--;
      x[il]--;
      sum1[il] += base1[il];
      sum2[il] += base2[il];
      for (int ip = il + 1; ip <= N; ++ip) {
        x[ip] = dim[ip] - 1;
        sum1[ip] = sum1[ip - 1];
        sum2[ip] = sum2[ip - 1];
      }
      index1 = sum1[N];
      index2 = sum2[N];
    }
  };

  /** Functions for Tensor class **/
  /**
     @brief Self contraction

     Z[ix0] = X[ix0][i][i], pair={1,2}={ix1,ix2}
     @param[out] Z Result of self contraction
     @param[in] X Original tensor
     @param[in] pair Array of pair positions for self contractions
     @param[in] num Number of pairs of self contractions
  */
  template<typename C> void self_contraction(Tensor<C>& Z, const Tensor<C>& X, const int* pair, const int num){
    Forall xx;
    Forall zz;

    std::vector<int> base_x(X.rank());
    int xu = 1;
    for (int i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
    }
    std::vector<int> flag_x(X.rank(), -1);
    for (int i = 0; i < num; ++i) {
      flag_x[pair[2 * i]] = i;
      flag_x[pair[2 * i + 1]] = i;
    }

    int rank = X.rank() - 2 * num;
    std::vector<int> dim(rank);
    xu = 1;
    for (int ip = 0, i = 0; i < X.rank(); ++i) {
      if (flag_x[i] == -1) {
        xx.push_back(xu, base_x[i], X.dimension(i));
        dim[ip++] = X.dimension(i);
        xu *= X.dimension(i);
      } else if (flag_x[i] >= 0) {
        int j = flag_x[i];
        int xdim = X.dimension( pair[2 * j] );
        if (xdim > X.dimension( pair[2 * j + 1] )) xdim = X.dimension( pair[2 * j + 1] );
        zz.push_back(base_x[pair[2 * j]], base_x[pair[2 * j + 1]], xdim);
        flag_x[pair[2 * j]] = -2;
        flag_x[pair[2 * j + 1]] = -2;
      }
    }
    Z.resize(rank, &(dim[0]));
    // Calculation
    for (xx.start(); xx.check(); xx.next()) {
      C sum = 0;
      for (zz.start(); zz.check(); zz.next())
        sum +=  X.element(xx.index2 + zz.index1 + zz.index2);
      Z[xx.index1] = sum;
    }
  }
  /**
     @brief Self contraction

     Z[ix0] = X[ix0][i][i], pair={1,2}={ix1,ix2}
     @param[out] Z Result of self contraction
     @param[in] X Original tensor
     @param[in] pairs Array of pair of positions for self contractions
  */
  template<typename C> void self_contraction(Tensor<C>& Z, const Tensor<C>& X, const std::vector< std::pair<int, int> > &pairs){
    Forall xx;
    Forall zz;
    int num = pairs.size();
    std::vector<int> base_x(X.rank());
    int xu = 1;
    for (int i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
    }
    std::vector<int> flag_x(X.rank(), -1);
    for (int i = 0; i < num; ++i) {
      flag_x[pairs[i].first] = i;
      flag_x[pairs[i].second] = i;
    }

    int rank = X.rank() - 2 * num;
    std::vector<int> dim(rank);
    xu = 1;
    for (int ip = 0, i = 0; i < X.rank(); ++i) {
      if (flag_x[i] == -1) {
        xx.push_back(xu, base_x[i], X.dimension(i));
        dim[ip++] = X.dimension(i);
        xu *= X.dimension(i);
      } else if (flag_x[i] >= 0) {
        int j = flag_x[i];
        int xdim = X.dimension( pairs[j].first );
        if (xdim > X.dimension( pairs[j].second )) xdim = X.dimension( pairs[j].second );
        zz.push_back(base_x[pairs[j].first], base_x[pairs[j].second], xdim);
        flag_x[pairs[j].first] = -2;
        flag_x[pairs[j].second] = -2;
      }
    }
    Z.resize(dim);
    // Calculation
    for (xx.start(); xx.check(); xx.next()) {
      C sum = 0;
      for (zz.start(); zz.check(); zz.next())
        sum +=  X.element(xx.index2 + zz.index1 + zz.index2);
      Z[xx.index1] = sum;
    }
  }

  /**
     @brief Make a subtensor

     Z[ix0] = X[ix0][x1][x2], value={-1,x1,x2}
     @param[out] Z Subtensor
     @param[in] X Original tensor
     @param[in] value Array of fixed value of indexes (The index for value[index]=-1 is remained).
  */
  template<typename C> void sub_tensor(Tensor<C>& Z, const Tensor<C>& X, const int* value){
    std::vector<int> dim(X.rank());
    Forall xx;
    int base = 0;
    int num = 0;
    int xu = 1;
    int xu0 = 1;
    for (int i = 0; i < X.rank(); ++i) {
      int xdim = X.dimension(i);
      if (value[i] == -1) {
        xx.push_back(xu0, xu, xdim);
        dim[num++] = xdim;
        xu0 *= xdim;
      } else
        base += xu * value[i];
      xu *= xdim;
    }
    // setup Z tensor
    Z.resize(num, &(dim[0]));
    // Calculation
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index1] =  X.element(xx.index2 + base);
  }

  /**
     @brief Make a subtensor

     Z[ix0] = X[ix0][x1][x2], value={-1,x1,x2}
     @param[out] Z Subtensor
     @param[in] X Original tensor
     @param[in] value Array of fixed value of indexes (The index for value[index]=-1 is remained).
  */
  template<typename C> void sub_tensor(Tensor<C>& Z, const Tensor<C>& X, const std::vector<int> &value){
    std::vector<int> dim(X.rank());
    Forall xx;
    int base = 0;
    int num = 0;
    int xu = 1;
    int xu0 = 1;
    for (int i = 0; i < X.rank(); ++i) {
      int xdim = X.dimension(i);
      if (value[i] == -1) {
        xx.push_back(xu0, xu, xdim);
        dim[num++] = xdim;
        xu0 *= xdim;
      } else
        base += xu * value[i];
      xu *= xdim;
    }
    // setup Z tensor
    Z.resize(num, &(dim[0]));
    // Calculation
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index1] =  X.element(xx.index2 + base);
  }

  /**
     @brief Rearrangement of tensor's indexes

     Z[ix[0]][ix[1]][ix[2]] = X[ix[value[0]]] [ix[value[1]]] [ix[value[2]]]\n
     The upper limits of value of i-th original index is dim[i]
     (ix[value[i]] < dim[i]).
     If dim is null (default), then there is no upper limit.
     @param[out] Z Tensor rearranged tensor's indexes
     @param[in] X Original tensor
     @param[in] value Array of new positions of original indexes
     @param[in] xdim Upper limits of values of indexes
  */
  template<typename C>  void rearrange(Tensor<C>& Z, const Tensor<C>& X, const int* value, const int* xdim = 0){
    std::vector<int> dim(X.rank());
    if (xdim == 0)
      for (int i = 0; i < X.rank(); ++i) dim[i] = X.dimension(i);
    else
      for (int i = 0; i < X.rank(); ++i) dim[i] = xdim[i];
    std::vector<int> base_x(X.rank());
    std::vector<int> base_x0(X.rank());
    std::vector<int> newdim(X.rank());
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
      newdim[value[i]] = dim[i];
    }
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x0[i] = xu;
      xu *= newdim[i];
    }
    Z.resize(X.rank(), &(newdim[0]));
    Forall xx;
    for (int i = 0; i < X.rank(); ++i)
      xx.push_back(base_x0[value[i]], base_x[i], dim[i]);
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index1] = X.element(xx.index2);
  }

  /**
     @brief Rearrangement of tensor's indexes

     Z[ix[0]][ix[1]][ix[2]] = X[ix[value[0]]] [ix[value[1]]] [ix[value[2]]]\n
     The upper limits of value of i-th original index is dim[i]
     (ix[value[i]] < dim[i]).
     If dim is null (default), then there is no upper limit.
     @param[out] Z Tensor rearranged tensor's indexes
     @param[in] X Original tensor
     @param[in] value Array of new positions of original indexes
  */
  template<typename C>  void rearrange(Tensor<C>& Z, const Tensor<C>& X, const std::vector<int> &value){
    std::vector<int> base_x(X.rank());
    std::vector<int> base_x0(X.rank());
    std::vector<int> newdim(X.rank());
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
      newdim[value[i]] = X.dimension(i);
    }
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x0[i] = xu;
      xu *= newdim[i];
    }
    Z.resize(newdim);
    Forall xx;
    for (int i = 0; i < X.rank(); ++i)
      xx.push_back(base_x0[value[i]], base_x[i], X.dimension(i));
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index1] = X.element(xx.index2);
  }

  /**
     @brief Rearrangement of tensor's indexes

     Z[ix[0]][ix[1]][ix[2]] = X[ix[value[0]]] [ix[value[1]]] [ix[value[2]]]\n
     The upper limits of value of i-th original index is dim[i]
     (ix[value[i]] < dim[i]).
     If dim is null (default), then there is no upper limit.
     @param[out] Z Tensor rearranged tensor's indexes
     @param[in] X Original tensor
     @param[in] value Array of new positions of original indexes
     @param[in] dim Upper limits of values of indexes
  */
  template<typename C>  void rearrange(Tensor<C>& Z, const Tensor<C>& X, const std::vector<int> &value, const std::vector<int> &dim){
    std::vector<int> base_x(X.rank());
    std::vector<int> base_x0(X.rank());
    std::vector<int> newdim(X.rank());
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
      newdim[value[i]] = dim[i];
    }
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x0[i] = xu;
      xu *= newdim[i];
    }
    Z.resize(newdim);
    Forall xx;
    for (int i = 0; i < X.rank(); ++i)
      xx.push_back(base_x0[value[i]], base_x[i], dim[i]);
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index1] = X.element(xx.index2);
  }

  /**
     @brief Swap of the first index and another index

     f(x,ip) = if x is ip then 0 else if x is 0 then ip else x\n
     Z[ix[0]][ix[1]][ix[2]] = X[ix[f(0,ip)]] [ix[f(1,ip)]] [ix[f(2,ip)]]\n
     @param[out] Z Tensor
     @param[in] X Tensor
     @param[in] ip Swapped index
  */
  template<typename C> void swap(Tensor<C>& Z, const Tensor<C>& X, const int ip){
    if (ip == 0)
      Z.copy(X);
    else{
      std::vector<int> newdim(X.rank());
      newdim[0] = X.dimension(ip);
      int na = 1;
      int base_a = X.dimension(0);
      int base_a_p = X.dimension(ip);
      for (int i = 1; i < ip; ++i) {
        na *= X.dimension(i);
        newdim[i] = X.dimension(i);
      }
      newdim[ip] = X.dimension(0);
      int base_ip = base_a * na;
      int base_ip_p = base_a_p * na;
      int base_b = base_ip * X.dimension(ip);
      int nb = 1;
      for (int i = ip + 1; i < X.rank(); ++i) {
        nb *= X.dimension(i);
        newdim[i] = X.dimension(i);
      }
      Z.resize(X.rank(), &(newdim[0]));

      for (int ib = 0, cb = 0; ib < nb; ++ib) {
        for (int ia = 0, ca = 0, ca_p = 0; ia < na; ++ia) {
          for (int i1 = 0, c1 = 0; i1 < X.dimension(ip); ++i1) {
            for (int i0 = 0, c0_p = 0; i0 < X.dimension(0); ++i0) {
              Z[i1 + ca_p + c0_p + cb] = X[i0 + ca + c1 + cb];
              c0_p += base_ip_p;
            }
            c1 += base_ip;
          }
          ca += base_a;
          ca_p += base_a_p;
        }
        cb += base_b;
      }
    }
  }

  /**
     @brief Change an index as the first index

     Z[ix[ip]][ix[0]][ix[1]]... = X[ix[0]][ix[1]]...
     @param[out] Z Tensor
     @param[in] X Tensor
     @param[in] ip Picked-up index
*/
  template<typename C> void pick(Tensor<C>& Z, const Tensor<C>& X, const int ip){
    if (ip == 0)
      Z.copy(X);
    else{
      std::vector<int> newdim(X.rank());
      newdim[0] = X.dimension(ip);
      int na = 1;
      int base_a_p = X.dimension(ip);
      for (int i = 0; i < ip; ++i) {
        na *= X.dimension(i);
        newdim[i + 1] = X.dimension(i);
      }
      int base_ip = na;
      int base_b = base_ip * X.dimension(ip);
      int nb = 1;
      for (int i = ip + 1; i < X.rank(); ++i) {
        nb *= X.dimension(i);
        newdim[i] = X.dimension(i);
      }
      Z.resize(X.rank(), &(newdim[0]));

      for (int ib = 0, cb = 0; ib < nb; ++ib) {
        for (int ia = 0, ca_p = 0; ia < na; ++ia) {
          for (int i1 = 0, c1 = 0; i1 < X.dimension(ip); ++i1) {
            Z[i1 + ca_p + cb] = X.element(ia + c1 + cb);
            c1 += base_ip;
          }
          ca_p += base_a_p;
        }
        cb += base_b;
      }
    }
  }

  /**
     @brief Embed with addition

     Z[ix0][x1][x2] += X[ix0], value={-1,x1,x2}
     @param[in] Z Original tensor
     @param[out] Z  Result of addition of an embed tensor to original tensor
     @param[in] X Embed tensor
     @param[in] value Array of fixed values of indexes of original tensor. The index for value[index]=-1 is remained.
     @note Z must be well defined.
  */
  template<typename C> void embed_add(Tensor<C>& Z, const Tensor<C>& X, const int* value){
    Forall xx;
    int base = 0;
    int xu = 1;
    int xu0 = 1;
    for (int i = 0; i < Z.rank(); ++i) {
      int xdim = Z.dimension(i);
      if (value[i] == -1) {
        xx.push_back(xu0, xu, xdim);
        xu0 *= xdim;
      } else
        base += xu * value[i];
      xu *= xdim;
    }
    // Calculation
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index2 + base] += X[xx.index1];
  }

  /**
     @brief Embed with addition

     Z[ix0][x1][x2] += X[ix0], value={-1,x1,x2}
     @param[in] Z Original tensor
     @param[out] Z  Result of addition of an embed tensor to original tensor
     @param[in] X Embed tensor
     @param[in] value Array of fixed values of indexes of original tensor. The index for value[index]=-1 is remained.
     @note Z must be well defined.
  */
  template<typename C> void embed_add(Tensor<C>& Z, const Tensor<C>& X, const std::vector<int> &value){
    Forall xx;
    int base = 0;
    int xu = 1;
    int xu0 = 1;
    for (int i = 0; i < Z.rank(); ++i) {
      int xdim = Z.dimension(i);
      if (value[i] == -1) {
        xx.push_back(xu0, xu, xdim);
        xu0 *= xdim;
      } else
        base += xu * value[i];
      xu *= xdim;
    }
    // Calculation
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index2 + base] += X[xx.index1];
  }

  /**
     @brief Reduce dimensions of indices

     Z[i][j]... (0 <= i < value[0], 0 <= j < value[1], ...)\n
     if value[] is larger than a dimension of tensor's index, it is not reduced.
     @param[out] Z Output tensor
     @param[in] X Input tensor
     @param[in] value New dimensions of indices
  */
  template<typename C> void reduce(Tensor<C>& Z, const Tensor<C>& X, const int* value){
    Forall xx;
    int base_x = 1;
    int base_z = 1;
    std::vector<int> dim(X.rank());
    for (int i = 0; i < X.rank(); ++i) {
      if (value[i] < 0) {
        xx.push_back(base_x, base_z, X.dimension(i));
        base_x *= X.dimension(i);
        base_z *= X.dimension(i);
        dim[i] = X.dimension(i);
      }else{
        int dim0 = std::min(value[i], X.dimension(i));
        xx.push_back(base_x, base_z, dim0);
        base_x *= X.dimension(i);
        base_z *= dim0;
        dim[i] = dim0;
      }
    }
    Z.initialize(dim.size(), &(dim[0]));
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index2] = X.element(xx.index1);
  }

  /**
     @brief Reduce dimensions of indices

     Z[i][j]... (0 <= i < value[0], 0 <= j < value[1], ...)\n
     if value[] is larger than a dimension of tensor's index, it is not reduced.
     @param[out] Z Output tensor
     @param[in] X Input tensor
     @param[in] value New dimensions of indices
  */
  template<typename C> void reduce(Tensor<C>& Z, const Tensor<C>& X, const std::vector<int> &value){
    Forall xx;
    int base_x = 1;
    int base_z = 1;
    std::vector<int> dim(X.rank());
    for (int i = 0; i < X.rank(); ++i) {
      if (value[i] < 0) {
        xx.push_back(base_x, base_z, X.dimension(i));
        base_x *= X.dimension(i);
        base_z *= X.dimension(i);
        dim[i] = X.dimension(i);
      }else{
        int dim0 = std::min(value[i], X.dimension(i));
        xx.push_back(base_x, base_z, dim0);
        base_x *= X.dimension(i);
        base_z *= dim0;
        dim[i] = dim0;
      }
    }
    Z.initialize(dim);
    for (xx.start(); xx.check(); xx.next())
      Z[xx.index2] = X.element(xx.index1);
  }

  // Contraction
#ifdef HAVE_BLAS
  /**
     @brief Contraction of two real tensors
     @note This function can use BLAS. If you have LAPACK, please
     define HAVE_LAPACK. If and only if you have BLAS without LAPACK,
     please define HAVE_BLAS.

     Z[ix0][ix2][iy1] = X[ix0][i][ix2] Y[i][iy1], pair={1,0}={ix1,iy0}
     @param[out] Z Result of contraction (Z = X . Y)
     @param[in] X Tensor
     @param[in] Y Tensor
     @param[in] pair Array of index pairs for contractions
     @param[in] num Number of index pairs
     @param[in] conjugate Ignored
  */
  void contraction(Tensor<double>& Z, const Tensor<double>& X, const Tensor<double>& Y, const int* pair, const int num, const bool conjugate = false){
    std::vector<int> newdim(X.rank() + Y.rank() - 2 * num);
    std::vector<int> xvalue(X.rank());
    std::vector<int> xdim(X.rank());
    std::vector<int> yvalue(Y.rank());
    std::vector<int> ydim(Y.rank());
    std::vector<int> pair_x(X.rank(), -1);
    std::vector<int> pair_y(Y.rank(), -1);
    for (int i = 0; i < num; ++i) {
      pair_x[pair[2 * i]] = i;
      pair_y[pair[2 * i + 1]] = i;
    }
    int ip = 0;
    for (int i = 0; i < X.rank(); ++i) {
      xdim[i] = X.dimension(i);
      if (pair_x[i] == -1) {
        xvalue[i] = ip;
        newdim[ip] = xdim[i];
        ip++;
      } else {
        xvalue[i] = X.rank() - num + pair_x[i];
        if (Y.dimension(pair[2 * pair_x[i] + 1]) < xdim[i])
          xdim[i] = Y.dimension(pair[2 * pair_x[i] + 1]);
      }
    }
    int ip0 = ip;
    ip = 0;
    for (int i = 0; i < Y.rank(); ++i) {
      ydim[i] = Y.dimension(i);
      if (pair_y[i] == -1) {
        yvalue[i] = num + ip;
        ip++;
        newdim[ip0++] = ydim[i];
      } else {
        yvalue[i] = pair_y[i];
        if (X.dimension(pair[2 * pair_y[i]]) < ydim[i])
          ydim[i] = X.dimension(pair[2 * pair_y[i]]);
      }
    }
    Tensor<double> A0;
    rearrange(A0, X, &(xvalue[0]), &(xdim[0]));
    Tensor<double> B0;
    rearrange(B0, Y, &(yvalue[0]), &(ydim[0]));
    int M = 1;
    for (int i = 0; i < (A0.rank() - num); ++i)
      M *= A0.dimension(i);
    int K = 1;
    for (int i = (A0.rank() - num); i < A0.rank(); ++i)
      K *= A0.dimension(i);
    int LDB = 1;
    for (int i = 0; i < num; ++i)
      LDB *= B0.dimension(i);
    if (LDB < K) LDB = K;
    int N = 1;
    for (int i = num; i < B0.rank(); ++i)
      N *= B0.dimension(i);
    Z.resize(newdim.size(), &(newdim[0]));
    double c0 = 0e0;
    double c1 = 1e0;
    char transa = 'N';
    char transb = 'N';
    DGEMM(&transa, &transb, &M, &N, &K, &c1, &(A0[0]), &M, &(B0[0]), &LDB, &c0, &(Z[0]), &M);
  }
  /**
     @brief Contraction of two real tensors
     @note This function can use BLAS. If you have LAPACK, please
     define HAVE_LAPACK. If and only if you have BLAS without LAPACK,
     please define HAVE_BLAS.

     Z[ix0][ix2][iy1] = X[ix0][i][ix2] Y[i][iy1], pair={1,0}={ix1,iy0}
     @param[out] Z Result of contraction (Z = X . Y)
     @param[in] X Tensor
     @param[in] Y Tensor
     @param[in] pairs Array of index pairs for contractions
     @param[in] conjugate Ignored
  */
  void contraction(Tensor<double>& Z, const Tensor<double>& X, const Tensor<double>& Y, const std::vector< std::pair<int, int> > &pairs, const bool conjugate = false){
    int num = pairs.size();
    std::vector<int> newdim(X.rank() + Y.rank() - 2 * num);
    std::vector<int> xvalue(X.rank());
    std::vector<int> xdim(X.rank());
    std::vector<int> yvalue(Y.rank());
    std::vector<int> ydim(Y.rank());
    std::vector<int> pair_x(X.rank(), -1);
    std::vector<int> pair_y(Y.rank(), -1);
    for (int i = 0; i < num; ++i) {
      pair_x[pairs[i].first] = i;
      pair_y[pairs[i].second] = i;
    }
    int ip = 0;
    for (int i = 0; i < X.rank(); ++i) {
      xdim[i] = X.dimension(i);
      if (pair_x[i] == -1) {
        xvalue[i] = ip;
        newdim[ip] = xdim[i];
        ip++;
      } else {
        xvalue[i] = X.rank() - num + pair_x[i];
        if (Y.dimension(pairs[pair_x[i]].second) < xdim[i])
          xdim[i] = Y.dimension(pairs[pair_x[i]].second);
      }
    }
    int ip0 = ip;
    ip = 0;
    for (int i = 0; i < Y.rank(); ++i) {
      ydim[i] = Y.dimension(i);
      if (pair_y[i] == -1) {
        yvalue[i] = num + ip;
        ip++;
        newdim[ip0++] = ydim[i];
      } else {
        yvalue[i] = pair_y[i];
        if (X.dimension(pairs[pair_y[i]].first) < ydim[i])
          ydim[i] = X.dimension(pairs[pair_y[i]].first);
      }
    }
    Tensor<double> A0;
    rearrange(A0, X, xvalue, xdim);
    Tensor<double> B0;
    rearrange(B0, Y, yvalue, ydim);
    int M = 1;
    for (int i = 0; i < (A0.rank() - num); ++i)
      M *= A0.dimension(i);
    int K = 1;
    for (int i = (A0.rank() - num); i < A0.rank(); ++i)
      K *= A0.dimension(i);
    int LDB = 1;
    for (int i = 0; i < num; ++i)
      LDB *= B0.dimension(i);
    if (LDB < K) LDB = K;
    int N = 1;
    for (int i = num; i < B0.rank(); ++i)
      N *= B0.dimension(i);
    Z.resize(newdim);
    double c0 = 0e0;
    double c1 = 1e0;
    char transa = 'N';
    char transb = 'N';
    DGEMM(&transa, &transb, &M, &N, &K, &c1, &(A0[0]), &M, &(B0[0]), &LDB, &c0, &(Z[0]), &M);
  }

  /**
     @brief Contraction of two complex tensors
     @note This function can use BLAS. If you have LAPACK, please
     define HAVE_LAPACK. If and only if you have BLAS without LAPACK,
     please define HAVE_BLAS.

     Z[ix0][ix2][iy1] = X[ix0][i][ix2] Y[i][iy1], pair={1,0}={ix1,iy0}

     If conjugate = true,

     Z[ix0][ix2][iy1] = (*X[ix0][i][ix2]) Y[i][iy1], pair={1,0}={ix1,iy0}
     @param[out] Z Result of contraction (Z = X . Y or Z = X* . Y)
     @param[in] X Tensor
     @param[in] Y Tensor
     @param[in] pair Array of index pairs for contractions
     @param[in] num Number of index pairs
     @param[in] conjugate If conjugate is true, conjugated contraction.
  */
  void contraction(Tensor< double_complex >& Z, const Tensor< double_complex >& X, const Tensor< double_complex >& Y, const int* pair, const int num, const bool conjugate = false){
    std::vector<int> newdim(X.rank() + Y.rank() - 2 * num);
    std::vector<int> xvalue(X.rank());
    std::vector<int> xdim(X.rank());
    std::vector<int> yvalue(Y.rank());
    std::vector<int> ydim(Y.rank());
    std::vector<int> pair_x(X.rank(), -1);
    std::vector<int> pair_y(Y.rank(), -1);
    for (int i = 0; i < num; ++i) {
      pair_x[pair[2 * i]] = i;
      pair_y[pair[2 * i + 1]] = i;
    }
    int ip = 0;
    for (int i = 0; i < X.rank(); ++i) {
      xdim[i] = X.dimension(i);
      if (pair_x[i] == -1) {
        xvalue[i] = ip;
        newdim[ip] = xdim[i];
        ip++;
      } else {
        xvalue[i] = X.rank() - num + pair_x[i];
        if (Y.dimension(pair[2 * pair_x[i] + 1]) < xdim[i])
          xdim[i] = Y.dimension(pair[2 * pair_x[i] + 1]);
      }
    }
    int ip0 = ip;
    ip = 0;
    for (int i = 0; i < Y.rank(); ++i) {
      ydim[i] = Y.dimension(i);
      if (pair_y[i] == -1) {
        yvalue[i] = num + ip;
        ip++;
        newdim[ip0++] = ydim[i];
      } else {
        yvalue[i] = pair_y[i];
        if (X.dimension(pair[2 * pair_y[i]]) < ydim[i])
          ydim[i] = X.dimension(pair[2 * pair_y[i]]);
      }
    }
    Tensor< double_complex > A0;
    rearrange(A0, X, &(xvalue[0]), &(xdim[0]));
    Tensor< double_complex > B0;
    rearrange(B0, Y, &(yvalue[0]), &(ydim[0]));
    int M = 1;
    for (int i = 0; i < (A0.rank() - num); ++i)
      M *= A0.dimension(i);
    int K = 1;
    for (int i = (A0.rank() - num); i < A0.rank(); ++i)
      K *= A0.dimension(i);
    int LDB = 1;
    for (int i = 0; i < num; ++i)
      LDB *= B0.dimension(i);
    if (LDB < K) LDB = K;
    int N = 1;
    for (int i = num; i < B0.rank(); ++i)
      N *= B0.dimension(i);
    Z.resize(newdim.size(), &(newdim[0]));
    double_complex c0 = double_complex(0, 0);
    double_complex c1 = double_complex(1, 0);
    if (conjugate) {
      double_complex *p = &(A0[0]);
      int NSIZE = A0.size();
      for (int i = 0; i < NSIZE; ++i) {
        *p = conj(*p);
        p++;
      }
    }
    char transa = 'N';
    char transb = 'N';
    ZGEMM(&transa, &transb, &M, &N, &K, &c1, &(A0[0]), &M, &(B0[0]), &LDB, &c0, &(Z[0]), &M);
  }
  /**
     @brief Contraction of two complex tensors
     @note This function can use BLAS. If you have LAPACK, please
     define HAVE_LAPACK. If and only if you have BLAS without LAPACK,
     please define HAVE_BLAS.

     Z[ix0][ix2][iy1] = X[ix0][i][ix2] Y[i][iy1], pair={1,0}={ix1,iy0}

     If conjugate = true,

     Z[ix0][ix2][iy1] = (*X[ix0][i][ix2]) Y[i][iy1], pair={1,0}={ix1,iy0}
     @param[out] Z Result of contraction (Z = X . Y or Z = X* . Y)
     @param[in] X Tensor
     @param[in] Y Tensor
     @param[in] pairs Array of index pairs for contractions
     @param[in] conjugate If conjugate is true, conjugated contraction.
  */
  void contraction(Tensor< double_complex >& Z, const Tensor< double_complex >& X, const Tensor< double_complex >& Y, const std::vector< std::pair<int, int> > pairs, const bool conjugate = false){
    int num = pairs.size();
    std::vector<int> newdim(X.rank() + Y.rank() - 2 * num);
    std::vector<int> xvalue(X.rank());
    std::vector<int> xdim(X.rank());
    std::vector<int> yvalue(Y.rank());
    std::vector<int> ydim(Y.rank());
    std::vector<int> pair_x(X.rank(), -1);
    std::vector<int> pair_y(Y.rank(), -1);
    for (int i = 0; i < num; ++i) {
      pair_x[pairs[i].first] = i;
      pair_y[pairs[i].second] = i;
    }
    int ip = 0;
    for (int i = 0; i < X.rank(); ++i) {
      xdim[i] = X.dimension(i);
      if (pair_x[i] == -1) {
        xvalue[i] = ip;
        newdim[ip] = xdim[i];
        ip++;
      } else {
        xvalue[i] = X.rank() - num + pair_x[i];
        if (Y.dimension(pairs[pair_x[i]].second) < xdim[i])
          xdim[i] = Y.dimension(pairs[pair_x[i]].second);
      }
    }
    int ip0 = ip;
    ip = 0;
    for (int i = 0; i < Y.rank(); ++i) {
      ydim[i] = Y.dimension(i);
      if (pair_y[i] == -1) {
        yvalue[i] = num + ip;
        ip++;
        newdim[ip0++] = ydim[i];
      } else {
        yvalue[i] = pair_y[i];
        if (X.dimension(pairs[pair_y[i]].first) < ydim[i])
          ydim[i] = X.dimension(pairs[pair_y[i]].first);
      }
    }
    Tensor< double_complex > A0;
    rearrange(A0, X, xvalue, xdim);
    Tensor< double_complex > B0;
    rearrange(B0, Y, yvalue, ydim);
    int M = 1;
    for (int i = 0; i < (A0.rank() - num); ++i)
      M *= A0.dimension(i);
    int K = 1;
    for (int i = (A0.rank() - num); i < A0.rank(); ++i)
      K *= A0.dimension(i);
    int LDB = 1;
    for (int i = 0; i < num; ++i)
      LDB *= B0.dimension(i);
    if (LDB < K) LDB = K;
    int N = 1;
    for (int i = num; i < B0.rank(); ++i)
      N *= B0.dimension(i);
    Z.resize(newdim);
    double_complex c0 = double_complex(0, 0);
    double_complex c1 = double_complex(1, 0);
    if (conjugate) {
      double_complex *p = &(A0[0]);
      int NSIZE = A0.size();
      for (int i = 0; i < NSIZE; ++i) {
        *p = conj(*p);
        p++;
      }
    }
    char transa = 'N';
    char transb = 'N';
    ZGEMM(&transa, &transb, &M, &N, &K, &c1, &(A0[0]), &M, &(B0[0]), &LDB, &c0, &(Z[0]), &M);
  }

  /**
     @brief Frobenius norm of complex tensor
  */
  double norm(Tensor< double_complex >& Z){
    int n = Z.size();
    int incx = 1;
    return DZNRM2(&n, &(Z[0]), &incx);
  }

  /**
     @brief Frobenius norm of real tensor
  */
  double norm(Tensor< double >& Z){
    int n = Z.size();
    int incx = 1;
    return DNRM2(&n, &(Z[0]), &incx);
  }

  /**
     @brief Inner product of two complex tensors
  */
  double_complex inner_product(Tensor< double_complex >& A, Tensor< double_complex >& B){
    int n = A.size();
    int incx = 1;
    int incy = 1;
    return ZDOTC(&n, &(A[0]), &incx, &(B[0]), &incy);
  }

  /**
     @brief Inner product of two real tensors
  */
  double inner_product(Tensor< double >& A, Tensor< double >& B){
    int n = A.size();
    int incx = 1;
    int incy = 1;
    return DDOT(&n, &(A[0]), &incx, &(B[0]), &incy);
  }

#else
  // No BLAS

  /**
     @brief Contraction of two real tensors

     Z[ix0][ix2][iy1] = X[ix0][i][ix2] Y[i][iy1], pair={1,0}={ix1,iy0}
     @param[out] Z Result of contraction (Z = X . Y)
     @param[in] X Tensor
     @param[in] Y Tensor
     @param[in] pair Array of index pairs for contractions
     @param[in] num Number of index pairs
     @param[in] conjugate Ignored
  */
  void contraction(Tensor<double>& Z, const Tensor<double>& X, const Tensor<double>& Y, const int* pair, const int num, const bool conjugate = false){
    if (num == 0) {
      int rank = X.rank() + Y.rank();
      int* dim = new int[rank];
      int ip = 0;
      for (int i = 0; i < X.rank(); ++i)
        dim[ip++] = X.dimension(i);
      for (int i = 0; i < Y.rank(); ++i)
        dim[ip++] = Y.dimension(i);
      Z.resize(rank, dim);
      delete[] dim;
      for (int ix = 0; ix < X.size(); ++ix)
        for (int iy = 0; iy < Y.size(); ++iy)
          Z[ix + iy * X.size()] = X.element(ix) * Y.element(iy);
      return;
    }

    std::vector<int> base_x(X.rank(), 0);
    std::vector<int> base_y(Y.rank(), 0);
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
    }
    for (int xu = 1, i = 0; i < Y.rank(); ++i) {
      base_y[i] = xu;
      xu *= Y.dimension(i);
    }

    std::vector<bool> flag_x(X.rank(), true);
    std::vector<bool> flag_y(Y.rank(), true);
    for (int i = 0; i < num; ++i) {
      assert(X.dimension(pair[2 * i]) <= Y.dimension(pair[2 * i + 1]));
      flag_x[pair[2 * i]] = false;
      flag_y[pair[2 * i + 1]] = false;
    }

    std::vector<int> base_x0(X.rank() - num, 0);
    std::vector<int> base_y0(Y.rank() - num, 0);
    int xu = 1;
    for (int ip = 0, i = 0; i < X.rank(); ++i) {
      if (flag_x[i]) {
        base_x0[ip++] = xu;
        xu *= X.dimension(i);
      }
    }
    for (int ip = 0, i = 0; i < Y.rank(); ++i) {
      if (flag_y[i]) {
        base_y0[ip++] = xu;
        xu *= Y.dimension(i);
      }
    }

    Forall xx;
    Forall yy;
    for (int ip = 0, i = 0; i < X.rank(); ++i)
      if (flag_x[i])
        xx.push_back(base_x0[ip++], base_x[i], X.dimension(i));
    for (int ip = 0, i = 0; i < Y.rank(); ++i)
      if (flag_y[i])
        yy.push_back(base_y0[ip++], base_y[i], Y.dimension(i));
    Forall zz;
    for (int i = 0; i < num; ++i) {
      // Insert DELTA_MATRIX D(i,j)=\delta_{i,j}
      int xdim;
      if (X.dimension(pair[2 * i]) <= Y.dimension(pair[2 * i + 1]))
        xdim = X.dimension(pair[2 * i]);
      else
        xdim = Y.dimension(pair[2 * i + 1]);
      // zz.push_back(base_x[pair[2 * i]], base_y[pair[2 * i + 1]], X.dimension(pair[2 * i]));
      zz.push_back(base_x[pair[2 * i]], base_y[pair[2 * i + 1]], xdim);
    }

    // setup Z tensor
    int rank = X.rank() + Y.rank() - 2 * num;
    int* dim = new int[rank];
    int ip = 0;
    for (int i = 0; i < X.rank(); ++i)
      if (flag_x[i]) dim[ip++] = X.dimension(i);
    for (int i = 0; i < Y.rank(); ++i)
      if (flag_y[i]) dim[ip++] = Y.dimension(i);
    Z.resize(rank, dim);
    delete[] dim;
    // Calculation
    for (xx.start(); xx.check(); xx.next())
      for (yy.start(); yy.check(); yy.next()) {
        double sum = 0;
        for (zz.start(); zz.check(); zz.next())
          sum += X.element(xx.index2 + zz.index1) * Y.element(yy.index2 + zz.index2);
        Z[xx.index1 + yy.index1] = sum;
      }
  }
  /**
     @brief Contraction of two real tensors

     Z[ix0][ix2][iy1] = X[ix0][i][ix2] Y[i][iy1], pair={1,0}={ix1,iy0}
     @param[out] Z Result of contraction (Z = X . Y)
     @param[in] X Tensor
     @param[in] Y Tensor
     @param[in] pairs Array of index pairs for contractions
     @param[in] conjugate Ignored
  */
  void contraction(Tensor<double>& Z, const Tensor<double>& X, const Tensor<double>& Y, const std::vector< std::pair<int, int> > &pairs, const bool conjugate = false){
    int num = pairs.size();
    if (num == 0) {
      int rank = X.rank() + Y.rank();
      std::vector<int> dim(rank);
      int ip = 0;
      for (int i = 0; i < X.rank(); ++i)
        dim[ip++] = X.dimension(i);
      for (int i = 0; i < Y.rank(); ++i)
        dim[ip++] = Y.dimension(i);
      Z.resize(dim);
      for (int ix = 0; ix < X.size(); ++ix)
        for (int iy = 0; iy < Y.size(); ++iy)
          Z[ix + iy * X.size()] = X.element(ix) * Y.element(iy);
      return;
    }

    std::vector<int> base_x(X.rank(), 0);
    std::vector<int> base_y(Y.rank(), 0);
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
    }
    for (int xu = 1, i = 0; i < Y.rank(); ++i) {
      base_y[i] = xu;
      xu *= Y.dimension(i);
    }

    std::vector<bool> flag_x(X.rank(), true);
    std::vector<bool> flag_y(Y.rank(), true);
    for (int i = 0; i < num; ++i) {
      assert(X.dimension(pairs[i].first) <= Y.dimension(pairs[i].second));
      flag_x[pairs[i].first] = false;
      flag_y[pairs[i].second] = false;
    }

    std::vector<int> base_x0(X.rank() - num, 0);
    std::vector<int> base_y0(Y.rank() - num, 0);
    int xu = 1;
    for (int ip = 0, i = 0; i < X.rank(); ++i) {
      if (flag_x[i]) {
        base_x0[ip++] = xu;
        xu *= X.dimension(i);
      }
    }
    for (int ip = 0, i = 0; i < Y.rank(); ++i) {
      if (flag_y[i]) {
        base_y0[ip++] = xu;
        xu *= Y.dimension(i);
      }
    }

    Forall xx;
    Forall yy;
    for (int ip = 0, i = 0; i < X.rank(); ++i)
      if (flag_x[i])
        xx.push_back(base_x0[ip++], base_x[i], X.dimension(i));
    for (int ip = 0, i = 0; i < Y.rank(); ++i)
      if (flag_y[i])
        yy.push_back(base_y0[ip++], base_y[i], Y.dimension(i));
    Forall zz;
    for (int i = 0; i < num; ++i) {
      // Insert DELTA_MATRIX D(i,j)=\delta_{i,j}
      int xdim;
      if (X.dimension(pairs[i].first) <= Y.dimension(pairs[i].second))
        xdim = X.dimension(pairs[i].first);
      else
        xdim = Y.dimension(pairs[i].second);
      // zz.push_back(base_x[pair[2 * i]], base_y[pair[2 * i + 1]], X.dimension(pair[2 * i]));
      zz.push_back(base_x[pairs[i].first], base_y[pairs[i].second], xdim);
    }

    // setup Z tensor
    int rank = X.rank() + Y.rank() - 2 * num;
    std::vector<int> dim(rank);
    int ip = 0;
    for (int i = 0; i < X.rank(); ++i)
      if (flag_x[i]) dim[ip++] = X.dimension(i);
    for (int i = 0; i < Y.rank(); ++i)
      if (flag_y[i]) dim[ip++] = Y.dimension(i);
    Z.resize(dim);
    // Calculation
    for (xx.start(); xx.check(); xx.next())
      for (yy.start(); yy.check(); yy.next()) {
        double sum = 0;
        for (zz.start(); zz.check(); zz.next())
          sum += X.element(xx.index2 + zz.index1) * Y.element(yy.index2 + zz.index2);
        Z[xx.index1 + yy.index1] = sum;
      }
  }

  /**
     @brief Contraction of two complex tensors

     Z[ix0][ix2][iy1] = X[ix0][i][ix2] Y[i][iy1], pair={1,0}={ix1,iy0}

     If conjugate = true,

     Z[ix0][ix2][iy1] = (*X[ix0][i][ix2]) Y[i][iy1], pair={1,0}={ix1,iy0}
     @param[out] Z Result of contraction (Z = X . Y or Z = X* . Y)
     @param[in] X Tensor
     @param[in] Y Tensor
     @param[in] pair Array of index pairs for contractions
     @param[in] num Number of index pairs
     @param[in] conjugate If conjugate is true, conjugated contraction.
  */
  void contraction(Tensor<double_complex>& Z, const Tensor<double_complex>& X, const Tensor<double_complex>& Y, const int* pair, const int num, const bool conjugate = false){
    if (num == 0) {
      int rank = X.rank() + Y.rank();
      int* dim = new int[rank];
      int ip = 0;
      for (int i = 0; i < X.rank(); ++i)
        dim[ip++] = X.dimension(i);
      for (int i = 0; i < Y.rank(); ++i)
        dim[ip++] = Y.dimension(i);
      Z.resize(rank, dim);
      delete[] dim;
      for (int ix = 0; ix < X.size(); ++ix)
        for (int iy = 0; iy < Y.size(); ++iy)
          Z[ix * Y.size() + iy] = conj(X.element(ix)) * Y.element(iy);
      return;
    }

    std::vector<int> base_x(X.rank(), 0);
    std::vector<int> base_y(Y.rank(), 0);
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
    }
    for (int xu = 1, i = 0; i < Y.rank(); ++i) {
      base_y[i] = xu;
      xu *= Y.dimension(i);
    }

    std::vector<bool> flag_x(X.rank(), true);  // Contraction index number (not -1)
    std::vector<bool> flag_y(Y.rank(), true);  // Contraction index number (not -1)
    for (int i = 0; i < num; ++i) {
      assert(X.dimension(pair[2 * i]) == Y.dimension(pair[2 * i + 1]));
      flag_x[pair[2 * i]] = false;
      flag_y[pair[2 * i + 1]] = false;
    }

    std::vector<int> base_x0(X.rank() - num, 0);
    std::vector<int> base_y0(Y.rank() - num, 0);
    int xu = 1;
    for (int ip = 0, i = 0; i < X.rank(); ++i) {
      if (flag_x[i]) {
        base_x0[ip++] = xu;
        xu *= X.dimension(i);
      }
    }
    for (int ip = 0, i = 0; i < Y.rank(); ++i) {
      if (flag_y[i]) {
        base_y0[ip++] = xu;
        xu *= Y.dimension(i);
      }
    }

    Forall xx;
    Forall yy;
    for (int ip = 0, i = 0; i < X.rank(); ++i)
      if (flag_x[i])
        xx.push_back(base_x0[ip++], base_x[i], X.dimension(i));
    for (int ip = 0, i = 0; i < Y.rank(); ++i)
      if (flag_y[i])
        yy.push_back(base_y0[ip++], base_y[i], Y.dimension(i));
    Forall zz;
    for (int i = 0; i < num; ++i) {
      int xdim;
      if (X.dimension(pair[2 * i]) <= Y.dimension(pair[2 * i + 1]))
        xdim = X.dimension(pair[2 * i]);
      else
        xdim = Y.dimension(pair[2 * i + 1]);
      //      zz.push_back(base_x[pair[2 * i]], base_y[pair[2 * i + 1]], X.dimension(pair[2 * i]));
      zz.push_back(base_x[pair[2 * i]], base_y[pair[2 * i + 1]], xdim);
    }

    // setup Z tensor
    int rank = X.rank() + Y.rank() - 2 * num;
    int* dim = new int[rank];
    int ip = 0;
    for (int i = 0; i < X.rank(); ++i)
      if (flag_x[i]) dim[ip++] = X.dimension(i);
    for (int i = 0; i < Y.rank(); ++i)
      if (flag_y[i]) dim[ip++] = Y.dimension(i);
    Z.resize(rank, dim);
    delete[] dim;
    // Calculation
    if (conjugate) {
      for (xx.start(); xx.check(); xx.next())
        for (yy.start(); yy.check(); yy.next()) {
          double_complex sum = 0;
          for (zz.start(); zz.check(); zz.next())
            sum += conj(X.element(xx.index2 + zz.index1)) * Y.element(yy.index2 + zz.index2);
          Z[xx.index1 + yy.index1] = sum;
        }
    }else{
      for (xx.start(); xx.check(); xx.next())
        for (yy.start(); yy.check(); yy.next()) {
          double_complex sum = 0;
          for (zz.start(); zz.check(); zz.next())
            sum += X.element(xx.index2 + zz.index1) * Y.element(yy.index2 + zz.index2);
          Z[xx.index1 + yy.index1] = sum;
        }
    }
  }
  /**
     @brief Contraction of two complex tensors

     Z[ix0][ix2][iy1] = X[ix0][i][ix2] Y[i][iy1], pair={1,0}={ix1,iy0}

     If conjugate = true,

     Z[ix0][ix2][iy1] = (*X[ix0][i][ix2]) Y[i][iy1], pair={1,0}={ix1,iy0}
     @param[out] Z Result of contraction (Z = X . Y or Z = X* . Y)
     @param[in] X Tensor
     @param[in] Y Tensor
     @param[in] pairs Array of index pairs for contractions
     @param[in] conjugate If conjugate is true, conjugated contraction.
  */
  void contraction(Tensor<double_complex>& Z, const Tensor<double_complex>& X, const Tensor<double_complex>& Y, const std::vector< std::pair<int, int> > pairs, const bool conjugate = false){
    int num = pairs.size();
    if (num == 0) {
      int rank = X.rank() + Y.rank();
      std::vector<int> dim(rank);
      int ip = 0;
      for (int i = 0; i < X.rank(); ++i)
        dim[ip++] = X.dimension(i);
      for (int i = 0; i < Y.rank(); ++i)
        dim[ip++] = Y.dimension(i);
      Z.resize(dim);
      for (int ix = 0; ix < X.size(); ++ix)
        for (int iy = 0; iy < Y.size(); ++iy)
          Z[ix * Y.size() + iy] = conj(X.element(ix)) * Y.element(iy);
      return;
    }

    std::vector<int> base_x(X.rank(), 0);
    std::vector<int> base_y(Y.rank(), 0);
    for (int xu = 1, i = 0; i < X.rank(); ++i) {
      base_x[i] = xu;
      xu *= X.dimension(i);
    }
    for (int xu = 1, i = 0; i < Y.rank(); ++i) {
      base_y[i] = xu;
      xu *= Y.dimension(i);
    }

    std::vector<bool> flag_x(X.rank(), true);  // Contraction index number (not -1)
    std::vector<bool> flag_y(Y.rank(), true);  // Contraction index number (not -1)
    for (int i = 0; i < num; ++i) {
      assert(X.dimension(pairs[i].first) == Y.dimension(pairs[i].second));
      flag_x[pairs[i].first] = false;
      flag_y[pairs[i].second] = false;
    }

    std::vector<int> base_x0(X.rank() - num, 0);
    std::vector<int> base_y0(Y.rank() - num, 0);
    int xu = 1;
    for (int ip = 0, i = 0; i < X.rank(); ++i) {
      if (flag_x[i]) {
        base_x0[ip++] = xu;
        xu *= X.dimension(i);
      }
    }
    for (int ip = 0, i = 0; i < Y.rank(); ++i) {
      if (flag_y[i]) {
        base_y0[ip++] = xu;
        xu *= Y.dimension(i);
      }
    }

    Forall xx;
    Forall yy;
    for (int ip = 0, i = 0; i < X.rank(); ++i)
      if (flag_x[i])
        xx.push_back(base_x0[ip++], base_x[i], X.dimension(i));
    for (int ip = 0, i = 0; i < Y.rank(); ++i)
      if (flag_y[i])
        yy.push_back(base_y0[ip++], base_y[i], Y.dimension(i));
    Forall zz;
    for (int i = 0; i < num; ++i) {
      int xdim;
      if (X.dimension(pairs[i].first) <= Y.dimension(pairs[i].second))
        xdim = X.dimension(pairs[i].first);
      else
        xdim = Y.dimension(pairs[i].second);
      //      zz.push_back(base_x[pair[2 * i]], base_y[pair[2 * i + 1]], X.dimension(pair[2 * i]));
      zz.push_back(base_x[pairs[i].first], base_y[pairs[i].second], xdim);
    }

    // setup Z tensor
    int rank = X.rank() + Y.rank() - 2 * num;
    std::vector<int> dim(rank);
    int ip = 0;
    for (int i = 0; i < X.rank(); ++i)
      if (flag_x[i]) dim[ip++] = X.dimension(i);
    for (int i = 0; i < Y.rank(); ++i)
      if (flag_y[i]) dim[ip++] = Y.dimension(i);
    Z.resize(dim);
    // Calculation
    if (conjugate) {
      for (xx.start(); xx.check(); xx.next())
        for (yy.start(); yy.check(); yy.next()) {
          double_complex sum = 0;
          for (zz.start(); zz.check(); zz.next())
            sum += conj(X.element(xx.index2 + zz.index1)) * Y.element(yy.index2 + zz.index2);
          Z[xx.index1 + yy.index1] = sum;
        }
    }else{
      for (xx.start(); xx.check(); xx.next())
        for (yy.start(); yy.check(); yy.next()) {
          double_complex sum = 0;
          for (zz.start(); zz.check(); zz.next())
            sum += X.element(xx.index2 + zz.index1) * Y.element(yy.index2 + zz.index2);
          Z[xx.index1 + yy.index1] = sum;
        }
    }
  }

  /**
     @brief Frobenius norm of complex tensor
  */
  double norm(Tensor< double_complex >& Z){
    double_complex *p = &(Z[0]);
    double x = 0;
    for (int i = 0; i < Z.size(); ++i) {
      x += (*p).real() * (*p).real() + (*p).imag() * (*p).imag();
      ++p;
    }
    return x;
  }

  /**
     @brief Frobenius norm of real tensor
  */
  double norm(Tensor< double >& Z){
    double *p = &(Z[0]);
    double x = 0;
    for (int i = 0; i < Z.size(); ++i) {
      x += (*p) * (*p);
      ++p;
    }
    return x;
  }

  /**
     @brief Inner product of two complex tensors
  */
  double_complex inner_product(Tensor< double_complex >& A, Tensor< double_complex >& B){
    double_complex *pa = &(A[0]);
    double_complex *pb = &(B[0]);
    double_complex x = double_complex(0, 0);
    for (int i = 0; i < A.size(); ++i) {
      x += conj(*pa) * (*pb);
      ++pa;
      ++pb;
    }
    return x;
  }

  /**
     @brief Inner product of two real tensors
  */
  double inner_product(Tensor< double >& A, Tensor< double >& B){
    double *pa = &(A[0]);
    double *pb = &(B[0]);
    double x = 0e0;
    for (int i = 0; i < A.size(); ++i) {
      x += (*pa) * (*pb);
      ++pa;
      ++pb;
    }
    return x;
  }
#endif

#ifdef HAVE_LAPACK
  // Singular value decomposition
/**
 @brief Singular value decomposition of complex matrix A as A = U S (V)**H
 @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

 @param[in] m Number of rows of matrix A
 @param[in] n Number of colomns of matrix A
 @param[in] A m x n complex matrix
 @param[out] A destoried
 @param[out] U m x m unitary matrix
 @param[out] S min(m,n) positive vector
 @param[out] VH Hermitian conjugate of n x n unitary matrix V
*/
  int svd(int m, int n, double_complex A[], double_complex U[], double S[], double_complex VH[]){
    char jobu = 'A';
    char jobvt = 'A';
    int lda = m;
    int ldu = m;
    int ldvt = n;
    int lwork = -1;
    double_complex work_size;
    std::vector<double> rwork(5 * std::min(m, n));
    int info;
    ZGESVD(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VH, &ldvt, &work_size, &lwork, &(rwork[0]), &info);
    if (info == 0)
      lwork = static_cast<int>(work_size.real());
    else
      lwork = 2 * std::min(m, n) + std::max(m, n);
    std::vector<double_complex> work(lwork);
    ZGESVD(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VH, &ldvt, &(work[0]), &lwork, &(rwork[0]), &info);
    return info;
  }

/**
   @brief Singular value decomposition of complex matrix A as A[i][j] = U[i][k] S[k] VH[k][j]
 @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

 @param[in] A m x n complex matrix (Tensor of rank 2)
 @param[out] A destoried
 @param[out] U m x m unitary matrix (Tensor of rank 2)
 @param[out] S min(m,n) positive vector
 @param[out] VH Hermitian conjugate of n x n unitary matrix V (Tensor of rank 2)
 @note Tensor A is destroyed.
*/
  int svd(Tensor<double_complex> &A, Tensor<double_complex> &U, std::vector<double> &S, Tensor<double_complex> &VH){
    assert(A.rank() == 2);
    assert(A.dimension(0) <= A.dimension(1));
    U.set_size(2, A.dimension(0), A.dimension(0));
    VH.set_size(2, A.dimension(1), A.dimension(1));
    S.resize(A.dimension(0));
    return svd(A.dimension(0), A.dimension(1), &(A[0]), &(U[0]), &(S[0]), &(VH[0]));
  }

/**
   @brief Singular value decomposition of real matrix A as A = U S (V)**T
   @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

   @param[in] m Number of rows of matrix A
   @param[in] n Number of colomns of matrix A
   @param[in] A m x n real matrix
   @param[out] A destoried
   @param[out] U m x m orthogonal matrix
   @param[out] S min(m,n) positive vector
   @param[out] VT Transpose of n x n orthogonal matrix V
*/
  int svd(int m, int n, double A[], double U[], double S[], double VT[]){
    char jobu = 'A';
    char jobvt = 'A';
    int lda = m;
    int ldu = m;
    int ldvt = n;
    int lwork = -1;
    double work_size;
    int info;
    DGESVD(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, &work_size, &lwork, &info);
    if (info == 0)
      lwork = static_cast<int>(work_size);
    else
      lwork = std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n));
    std::vector<double> work(lwork);
    DGESVD(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, &(work[0]), &lwork, &info);
    return info;
  }

/**
   @brief Singular value decomposition of real matrix A as A[i][j] = U[i][k] S[k] VT[k][j]
   @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

   @param[in] A m x n real matrix (Tensor of rank 2)
   @param[out] A destoried
   @param[out] U m x m orthogonal matrix (Tensor of rank 2)
   @param[out] S min(m,n) positive vector
   @param[out] VT Transpose of n x n orthogonal matrix V (Tensor of rank 2)
   @note Tensor A is destroyed.
*/
  int svd(Tensor<double> &A, Tensor<double> &U, std::vector<double> &S, Tensor<double> &VT){
    assert(A.rank() == 2);
    assert(A.dimension(0) <= A.dimension(1));
    U.set_size(2, A.dimension(0), A.dimension(0));
    VT.set_size(2, A.dimension(1), A.dimension(1));
    S.resize(A.dimension(0));
    return svd(A.dimension(0), A.dimension(1), &(A[0]), &(U[0]), &(S[0]), &(VT[0]));
  }

/**
 @brief Singular value decomposition of complex matrix A (Num. of rows <= Num. of colomns) as A = U S (V)**H
 @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

 @param[in] m Number of rows of matrix A
 @param[in] n Number of colomns of matrix A
 @note  m <= n (Num. of rows <= Num. of colomns)
 @param[in] A m x n complex matrix, but m <= n
 @param[out] A destoried
 @param[out] U m x m unitary matrix
 @param[out] S m positive vector
 @param[out] VH first m rows of Hermitian conjugate of n x n unitary matrix V
*/
  int svd_r(int m, int n, double_complex A[], double_complex U[], double S[], double_complex VH[]){
    assert(m <= n);
    char jobu = 'A';
    char jobvt = 'S';
    int lda = m;
    int ldu = m;
    int ldvt = m;
    int lwork = -1;
    double_complex work_size;
    std::vector<double> rwork(5 * m);
    int info;
    ZGESVD(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VH, &ldvt, &work_size, &lwork, &(rwork[0]), &info);
    if (info == 0)
      lwork = static_cast<int>(work_size.real());
    else
      lwork = 2 * m + n;
    std::vector<double_complex> work(lwork);
    ZGESVD(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VH, &ldvt, &(work[0]), &lwork, &(rwork[0]), &info);
    return info;
  }

/**
   @brief Singular value decomposition of real matrix A  (Num. of rows <= Num. of colomns) as A = U S (V)**T
   @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

   @param[in] m Number of rows of matrix A
   @param[in] n Number of colomns of matrix A
   @note  m <= n (Num. of rows <= Num. of colomns)
   @param[in] A m x n matrix, but m <= n
   @param[out] A destoried
   @param[out] U m x m orthogonal matrix
   @param[out] S m positive vector
   @param[out] VT first m rows of transpose of n x n orthogonal matrix V
*/
  int svd_r(int m, int n, double A[], double U[], double S[], double VT[]){
    assert(m <= n);
    char jobu = 'A';
    char jobvt = 'S';
    int lda = m;
    int ldu = m;
    int ldvt = m;
    int lwork = -1;
    double work_size;
    int info;
    DGESVD(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, &work_size, &lwork, &info);
    if (info == 0)
      lwork = static_cast<int>(work_size);
    else
      lwork = std::max(3 * m + n, 5 * m);
    std::vector<double> work(lwork);
    DGESVD(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, &(work[0]), &lwork, &info);
    return info;
  }

  // Diagonalization

/**
   @brief Diagonalization of Hermitian matrix
   @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

   @param[in] n Number of rows and columns of Hermitian matrix
   @param[in] A n x n matrix, hermitian matrix (stored in the upper part of A)
   @param[out] A n eigenvectors
   @param[out] W n eigenvalues (ascending order)
*/
  int diagonalize(int n, double_complex A[], double W[]){
    char jobz = 'V';
    char uplo = 'U';
    int lda = n;
    int lwork = -1;
    double_complex work_size;
    int lrwork = -1;
    double rwork_size;
    int liwork = -1;
    int iwork_size;
    int info;
    ZHEEVD(&jobz, &uplo, &n, A, &lda, W, &work_size, &lwork, &rwork_size, &lrwork, &iwork_size, &liwork, &info);
    if (info == 0) {
      lwork = static_cast<int>(work_size.real());
      lrwork = rwork_size;
      liwork = iwork_size;
    }else{
      lwork = 2 * n + n * n;
      lrwork = 1 + 5 * n + 2 * n * n;
      liwork = 3 + 5 * n;
    }
    std::vector<double_complex> work(lwork);
    std::vector<double> rwork(lrwork);
    std::vector<int> iwork(liwork);
    ZHEEVD(&jobz, &uplo, &n, A, &lda, W, &(work[0]), &lwork, &(rwork[0]), &lrwork, &(iwork[0]), &liwork, &info);
    return info;
  }

/**
   @brief Diagonalization of real symmetric matrix
   @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

   @param[in] n Number of rows and columns of real symmetric matrix
   @param[in] A n x n matrix, real symmetric matrix (stored in the upper part of A)
   @param[out] A n eigenvectors
   @param[out] W n eigenvalues (ascending order)
*/
  int diagonalize(int n, double A[], double W[]){
    char jobz = 'V';
    char uplo = 'U';
    int lda = n;
    int lwork = -1;
    double work_size;
    int liwork = -1;
    int iwork_size;
    int info;
    DSYEVD(&jobz, &uplo, &n, A, &lda, W, &work_size, &lwork, &iwork_size, &liwork, &info);
    if (info == 0) {
      lwork = work_size;
      liwork = iwork_size;
    }else{
      lwork = 1 + 6 * n + 2 * n * n;
      liwork = 3 + 5 * n;
    }
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);
    DSYEVD(&jobz, &uplo, &n, A, &lda, W, &(work[0]), &lwork, &(iwork[0]), &liwork, &info);
    return info;
  }

/**
   @brief Diagonalization of Hermitian matrix (ascending order, only first m ones)
   @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

   @param[in] m
   @param[in] n Number of rows and columns of Hermitian matrix
   @param[in] A n x n matrix, hermitian matrix (stored in the upper part of A)
   @param[out] A destroyed
   @param[out] W n eigenvalues (ascending order, only first m ones)
   @param[out] Z n x m matrix, eigenvectors (only first m columns)
*/
  int diagonalize(int m, int n, double_complex A[], double W[], double_complex Z[]){
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int lda = n;
    double vl, vu;
    int il = 1;
    int iu = m;
    double abstol = -1e0;
    int ldz = n;
    std::vector<double> rwork(7 * n);
    std::vector<int> iwork(5 * n);
    std::vector<int> ifail(n);
    int info;
    int lwork = -1;
    double_complex work_size;
    ZHEEVX(&jobz, &range, &uplo, &n, A, &lda, &vl, &vu, &il, &iu, &abstol, &m, W, Z, &ldz, &work_size, &lwork, &(rwork[0]), &(iwork[0]), &(ifail[0]), &info);
    if (info == 0)
      lwork = static_cast<int>(work_size.real());
    else
      lwork = 2 * n - 1;
    std::vector<double_complex> work(lwork);
    ZHEEVX(&jobz, &range, &uplo, &n, A, &lda, &vl, &vu, &il, &iu, &abstol, &m, W, Z, &ldz, &(work[0]), &lwork, &(rwork[0]), &(iwork[0]), &(ifail[0]), &info);
    return info;
  }

/**
   @brief Diagonalization of real symmetric matrix (ascending order, only first m ones)
   @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

   @param[in] m
   @param[in] n Number of rows and columns of real symmetric matrix
   @param[in] A n x n matrix, real symmetric matrix (stored in the upper part of A)
   @param[out] A destroyed
   @param[out] W n eigenvalues (ascending order, only first m ones)
   @param[out] Z n x m matrix, eigenvectors (only first m columns)
*/
  int diagonalize(int m, int n, double A[], double W[], double Z[]){
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int lda = n;
    double vl, vu;
    int il = 1;
    int iu = m;
    double abstol = -1e0;
    int ldz = n;
    std::vector<double> rwork(7 * n);
    std::vector<int> iwork(5 * n);
    std::vector<int> ifail(n);
    int info;
    int lwork = -1;
    double work_size;
    DSYEVX(&jobz, &range, &uplo, &n, A, &lda, &vl, &vu, &il, &iu, &abstol, &m, W, Z, &ldz, &work_size, &lwork, &(iwork[0]), &(ifail[0]), &info);
    if (info == 0)
      lwork = work_size;
    else
      lwork = 8 * n;
    std::vector<double> work(lwork);
    DSYEVX(&jobz, &range, &uplo, &n, A, &lda, &vl, &vu, &il, &iu, &abstol, &m, W, Z, &ldz, &(work[0]), &lwork, &(iwork[0]), &(ifail[0]), &info);
    return info;
  }

  /**
     @brief Higher-order singular value decomposition of complex tensor
     @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

     A_{ijk...} = S_{abc...} * US[0]_{ia} * US[1]_{jb} * ...
     @param[in] A Complex tensor
     @param[out] US Array of unitary matrix (2-rank tensor)
     @param[out] S Core tensor
  */
  int hosvd(const Tensor<double_complex> &A, std::vector<Tensor<double_complex> > &US, Tensor<double_complex> &S){
    US.resize(A.rank());
    S.copy(A);
    int pair[2];
    pair[0] = 0;
    pair[1] = S.rank() - 1;
    for (int ip = (A.rank() - 1); ip >= 0; --ip) {
      Tensor<double_complex> Z;
      pick(Z, A, ip);
      Tensor<double_complex> C(2, Z.dimension(0)); // diagonalized matrix
#ifdef USE_DIRECTLY_LAPACK
      char jobu = 'A';
      char jobvt = 'N';
      int m = Z.dimension(0);
      int n = Z.size() / m;
      int lda = m;
      int ldu = m;
      int ldvt = n;
      int lwork = -1;
      double_complex work_size;
      std::vector<double> rwork(5 * std::min(m, n));
      int info;
      std::vector<double> LAMBDA(std::min(m, n));
      double_complex *null_vt;
      ZGESVD(&jobu, &jobvt, &m, &n, &(Z[0]), &lda, &(LAMBDA[0]), &(C[0]), &ldu, null_vt, &ldvt, &work_size, &lwork, &(rwork[0]), &info);
      if (info == 0)
        lwork = static_cast<int>(work_size.real());
      else
        lwork = 2 * std::min(m, n) + std::max(m, n);
      std::vector<double_complex> work(lwork);
      ZGESVD(&jobu, &jobvt, &m, &n, &(Z[0]), &lda, &(LAMBDA[0]), &(C[0]), &ldu, null_vt, &ldvt, &(work[0]), &lwork, &(rwork[0]), &info);
      if (info != 0) return info;
#else
      char uplo = 'U';
      char trans = 'N';
      int n = Z.dimension(0);
      int k = Z.size() / n;
      double alpha = 1e0;
      int lda = n;
      double beta = 0;
      int ldc = n;
      ZHERK(&uplo, &trans, &n, &k, &alpha, &(Z[0]), &lda, &beta, &(C[0]), &ldc);
      char jobz = 'V';
      std::vector<double> w(n);
      double_complex work_size;
      int lwork = -1;
      double rwork_size;
      int lrwork = -1;
      int iwork_size;
      int liwork = -1;
      int info;
      ZHEEVD(&jobz, &uplo, &n, &(C[0]), &lda, &(w[0]), &work_size, &lwork, &rwork_size, &lrwork, &iwork_size, &liwork, &info);
      if (info == 0) {
        lwork = static_cast<int>(work_size.real());
        lrwork = static_cast<int>(rwork_size);
        liwork = iwork_size;
      }else{
        lwork = 2 * n + n * n;
        lrwork = 1 + 5 * n + 2 * n * n;
        liwork = 3 + 5 * n;
      }
      std::vector<double_complex> work(lwork);
      std::vector<double> rwork(lrwork);
      std::vector<int> iwork(liwork);
      ZHEEVD(&jobz, &uplo, &n, &(C[0]), &lda, &(w[0]), &(work[0]), &lwork, &(rwork[0]), &lrwork, &(iwork[0]), &liwork, &info); // Eigenvalues in ascending order
      if (info != 0) return info;
      // Change eigenvalues in descending order
      int c0 = 0;
      int c1 = C.size() - n;
      int inc = 1;
      for (int i = 0; i < n / 2; ++i) {
        ZSWAP(&n, &(C[c0]), &inc, &(C[c1]), &inc);
        c0 += n;
        c1 -= n;
      }
#endif
      // Unitary tensor
      US[ip].copy(C);
      // Calculate a core tensor
      Tensor<double_complex> S0(S);
      contraction(S, C, S0, pair, 1, true);
    }
    return 0;
  }
  /**
     @brief Higher-order singular value decomposition of real tensor
     @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

     A_{ijk...} = S_{abc...} * US[0]_{ia} * US[1]_{jb} * ...
     @param[in] A Real tensor
     @param[out] US Array of orthogonal matrix (2-rank tensor)
     @param[out] S Core tensor
  */
  int hosvd(const Tensor<double> &A, std::vector<Tensor<double> > &US, Tensor<double> &S){
    US.resize(A.rank());
    S.copy(A);
    int pair[2];
    pair[0] = 0;
    pair[1] = S.rank() - 1;
    for (int ip = (A.rank() - 1); ip >= 0; --ip) {
      Tensor<double> D;
      pick(D, A, ip);
      Tensor<double> C(2, D.dimension(0)); // diagonalized matrix
#ifdef USE_DIRECTLY_LAPACK
      char jobu = 'A';
      char jobvt = 'N';
      int m = D.dimension(0);
      int n = D.size() / m;
      int lda = m;
      int ldu = m;
      int ldvt = n;
      int lwork = -1;
      double work_size;
      int info;
      std::vector<double> LAMBDA(std::min(m, n));
      double *null_vt;
      DGESVD(&jobu, &jobvt, &m, &n, &(D[0]), &lda, &(LAMBDA[0]), &(C[0]), &ldu, null_vt, &ldvt, &work_size, &lwork, &info);
      if (info == 0)
        lwork = static_cast<int>(work_size);
      else
        lwork = std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n));
      std::vector<double> work(lwork);
      DGESVD(&jobu, &jobvt, &m, &n, &(D[0]), &lda, &(LAMBDA[0]), &(C[0]), &ldu, null_vt, &ldvt, &(work[0]), &lwork, &info);
      if (info != 0) return info;
#else
      char uplo = 'U';
      char trans = 'N';
      int n = D.dimension(0);
      int k = D.size() / n;
      double alpha = 1e0;
      int lda = n;
      double beta = 0;
      int ldc = n;
      DSYRK(&uplo, &trans, &n, &k, &alpha, &(D[0]), &lda, &beta, &(C[0]), &ldc);
      char jobz = 'V';
      std::vector<double> w(n);
      double work_size;
      int lwork = -1;
      int iwork_size;
      int liwork = -1;
      int info;
      DSYEVD(&jobz, &uplo, &n, &(C[0]), &lda, &(w[0]), &work_size, &lwork, &iwork_size, &liwork, &info);
      if (info == 0) {
        lwork = static_cast<int>(work_size);
        liwork = iwork_size;
      }else{
        lwork = 1 + 6 * n + 2 * n * n;
        liwork = 3 + 5 * n;
      }
      std::vector<double> work(lwork);
      std::vector<int> iwork(liwork);
      DSYEVD(&jobz, &uplo, &n, &(C[0]), &lda, &(w[0]), &(work[0]), &lwork, &(iwork[0]), &liwork, &info); // Eigenvalues in ascending order
      // Change eigenvalues in descending order
      if (info != 0) return info;
      int c0 = 0;
      int c1 = C.size() - n;
      int inc = 1;
      for (int i = 0; i < n / 2; ++i) {
        DSWAP(&n, &(C[c0]), &inc, &(C[c1]), &inc);
        c0 += n;
        c1 -= n;
      }
#endif
      // Unitary tensor
      US[ip].copy(C);
      // Calculate a core tensor
      Tensor<double> S0(S);
      contraction(S, C, S0, pair, 1);
    }
    return 0;
  }

  /**
     @brief Higher-order singular value decomposition of complex tensor
     @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

     A_{ijk...} = S_{abc...} * US[0]_{ia} * US[1]_{jb} * ...
     @param[in] A Complex tensor
     @param[out] US Array of unitary matrix (2-rank tensor)
     @param[out] SS Array of singular value's vector for each indices
  */
  int hosvd(const Tensor<double_complex> &A, std::vector<Tensor<double_complex> > &US, std::vector< std::vector<double> > &SS){
    US.resize(A.rank());
    SS.resize(A.rank());
    for (int ip = (A.rank() - 1); ip >= 0; --ip) {
      Tensor<double_complex> Z;
      pick(Z, A, ip);
      Tensor<double_complex> C(2, Z.dimension(0)); // diagonalized matrix
#ifdef USE_DIRECTLY_LAPACK
      char jobu = 'A';
      char jobvt = 'N';
      int m = Z.dimension(0);
      int n = Z.size() / m;
      int lda = m;
      int ldu = m;
      int ldvt = n;
      int lwork = -1;
      double_complex work_size;
      std::vector<double> rwork(5 * std::min(m, n));
      int info;
      std::vector<double> LAMBDA(std::min(m, n));
      double_complex *null_vt;
      ZGESVD(&jobu, &jobvt, &m, &n, &(Z[0]), &lda, &(LAMBDA[0]), &(C[0]), &ldu, null_vt, &ldvt, &work_size, &lwork, &(rwork[0]), &info);
      if (info == 0)
        lwork = static_cast<int>(work_size.real());
      else
        lwork = 2 * std::min(m, n) + std::max(m, n);
      std::vector<double_complex> work(lwork);
      ZGESVD(&jobu, &jobvt, &m, &n, &(Z[0]), &lda, &(LAMBDA[0]), &(C[0]), &ldu, null_vt, &ldvt, &(work[0]), &lwork, &(rwork[0]), &info);
      if (info != 0) return info;
      SS[ip] = LAMBDA;
#else
      char uplo = 'U';
      char trans = 'N';
      int n = Z.dimension(0);
      int k = Z.size() / n;
      double alpha = 1e0;
      int lda = n;
      double beta = 0;
      int ldc = n;
      ZHERK(&uplo, &trans, &n, &k, &alpha, &(Z[0]), &lda, &beta, &(C[0]), &ldc);
      char jobz = 'V';
      std::vector<double> w(n);
      double_complex work_size;
      int lwork = -1;
      double rwork_size;
      int lrwork = -1;
      int iwork_size;
      int liwork = -1;
      int info;
      ZHEEVD(&jobz, &uplo, &n, &(C[0]), &lda, &(w[0]), &work_size, &lwork, &rwork_size, &lrwork, &iwork_size, &liwork, &info);
      if (info == 0) {
        lwork = static_cast<int>(work_size.real());
        lrwork = static_cast<int>(rwork_size);
        liwork = iwork_size;
      }else{
        lwork = 2 * n + n * n;
        lrwork = 1 + 5 * n + 2 * n * n;
        liwork = 3 + 5 * n;
      }
      std::vector<double_complex> work(lwork);
      std::vector<double> rwork(lrwork);
      std::vector<int> iwork(liwork);
      ZHEEVD(&jobz, &uplo, &n, &(C[0]), &lda, &(w[0]), &(work[0]), &lwork, &(rwork[0]), &lrwork, &(iwork[0]), &liwork, &info); // Eigenvalues in ascending order
      if (info != 0) return info;
      // Change eigenvalues in descending order
      int c0 = 0;
      int c1 = C.size() - n;
      int inc = 1;
      for (int i = 0; i < n / 2; ++i) {
        ZSWAP(&n, &(C[c0]), &inc, &(C[c1]), &inc);
        c0 += n;
        c1 -= n;
      }
      SS[ip] = w;
      reverse(SS[ip].begin(), SS[ip].end());
#endif
      // Unitary tensor
      US[ip].copy(C);
    }
    return 0;
  }
  /**
     @brief Higher-order singular value decomposition of real tensor
     @note This function uses LAPACK. To use this function, you need define a macro HAVE_LAPACK.

     A_{ijk...} = S_{abc...} * US[0]_{ia} * US[1]_{jb} * ...
     @param[in] A Real tensor
     @param[out] US Array of orthogonal matrix (2-rank tensor)
     @param[out] SS Array of singular value's vector for each indices
  */
  int hosvd(const Tensor<double> &A, std::vector<Tensor<double> > &US, std::vector< std::vector<double> > &SS){
    US.resize(A.rank());
    SS.resize(A.rank());
    for (int ip = (A.rank() - 1); ip >= 0; --ip) {
      Tensor<double> D;
      pick(D, A, ip);
      Tensor<double> C(2, D.dimension(0)); // diagonalized matrix
#ifdef USE_DIRECTLY_LAPACK
      char jobu = 'A';
      char jobvt = 'N';
      int m = D.dimension(0);
      int n = D.size() / m;
      int lda = m;
      int ldu = m;
      int ldvt = n;
      int lwork = -1;
      double work_size;
      int info;
      std::vector<double> LAMBDA(std::min(m, n));
      double *null_vt;
      DGESVD(&jobu, &jobvt, &m, &n, &(D[0]), &lda, &(LAMBDA[0]), &(C[0]), &ldu, null_vt, &ldvt, &work_size, &lwork, &info);
      if (info == 0)
        lwork = static_cast<int>(work_size);
      else
        lwork = std::max(3 * std::min(m, n) + std::max(m, n), 5 * std::min(m, n));
      std::vector<double> work(lwork);
      DGESVD(&jobu, &jobvt, &m, &n, &(D[0]), &lda, &(LAMBDA[0]), &(C[0]), &ldu, null_vt, &ldvt, &(work[0]), &lwork, &info);
      if (info != 0) return info;
      SS[ip] = LAMBDA;
#else
      char uplo = 'U';
      char trans = 'N';
      int n = D.dimension(0);
      int k = D.size() / n;
      double alpha = 1e0;
      int lda = n;
      double beta = 0;
      int ldc = n;
      DSYRK(&uplo, &trans, &n, &k, &alpha, &(D[0]), &lda, &beta, &(C[0]), &ldc);
      char jobz = 'V';
      std::vector<double> w(n);
      double work_size;
      int lwork = -1;
      int iwork_size;
      int liwork = -1;
      int info;
      DSYEVD(&jobz, &uplo, &n, &(C[0]), &lda, &(w[0]), &work_size, &lwork, &iwork_size, &liwork, &info);
      if (info == 0) {
        lwork = static_cast<int>(work_size);
        liwork = iwork_size;
      }else{
        lwork = 1 + 6 * n + 2 * n * n;
        liwork = 3 + 5 * n;
      }
      std::vector<double> work(lwork);
      std::vector<int> iwork(liwork);
      DSYEVD(&jobz, &uplo, &n, &(C[0]), &lda, &(w[0]), &(work[0]), &lwork, &(iwork[0]), &liwork, &info); // Eigenvalues in ascending order
      // Change eigenvalues in descending order
      if (info != 0) return info;
      int c0 = 0;
      int c1 = C.size() - n;
      int inc = 1;
      for (int i = 0; i < n / 2; ++i) {
        DSWAP(&n, &(C[c0]), &inc, &(C[c1]), &inc);
        c0 += n;
        c1 -= n;
      }
      SS[ip] = w;
      reverse(SS[ip].begin(), SS[ip].end());
#endif
      // Unitary tensor
      US[ip].copy(C);
    }
    return 0;
  }
#endif

}
#endif  // _MY_TENSOR_CALCULATION_TOOLKIT_HPP_
