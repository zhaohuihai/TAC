// google c++ style: http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml

//#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <boost/shared_ptr.hpp>

#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#include <mkl.h>

#include "tensor.hpp"

//========================================================================================
//int main(int argc, char **argv)
//{
//    int rank, size;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//    printf("I am %d of %d\n", rank + 1, size);
//    MPI_Finalize();
//
//return 0;
//}

//class myClass
//{
//public:
//  double data[10] ;
//  myClass(double a)
//  {
//    for (int i = 0; i < 10; i ++)
//      {
//	data[i] = double(i) * a ;
//      }
//  }
//
//  void display()
//  {
//    for (int i = 0; i < 10; i ++)
//      {
//	std::cout << data[i] << std::endl ;
//      }
//  }
//};
//
//int main()
//{
//  boost::shared_ptr<myClass> ptr;
//
//  std::vector< boost::shared_ptr<myClass> > acc ;
//
//  boost::shared_ptr<myClass> temp(new myClass(0.1)) ;
//
//  acc.push_back(temp) ;
//
//  acc.at(0)->display() ;
//
//  return 0 ;
//}


//#define    NUM_THREADS    8
//
//void *PrintHello(void *args)
//{
//    int thread_arg;
////    sleep(1);
//    thread_arg = (int)args;
//    printf("Hello from thread %d\n", thread_arg);
//    return NULL;
//}
//
//int main(void)
//{
//    int rc,t;
//    pthread_t thread[NUM_THREADS];
//
//    for( t = 0; t < NUM_THREADS; t++)
//    {
////        printf("Creating thread %d\n", t);
//        rc = pthread_create(&thread[t], NULL, PrintHello, (void *)t);
//        if (rc)
//        {
//            printf("ERROR; return code is %d\n", rc);
//            return EXIT_FAILURE;
//        }
//    }
////    for( t = 0; t < NUM_THREADS; t++)
////        pthread_join(thread[t], NULL);
//    return EXIT_SUCCESS;
//}

int main(void)
{
//	 start measuring time cost

	int D = 2000 ;
	Tensor<double> A(4, 3) ;
	A.arithmeticSequence(0, 1) ;
	Tensor<double> B = A + 1.0 ;
//	B.arithmeticSequence(1, 2) ;

	Tensor<int> P(3) ;
	P[0] = 2 ;
	P[1] = 0 ;
	P[2] = 1 ;

	double time_start = dsecnd() ;

	for (int j = 0; j < 100; j ++)
	{
		for (int i = 0; i < 1; i ++)
		{
//			A = spitVector(A, 1, B) ;
		}
	}

	TensorArray<double> T(2) ;
	T(0) = A ;
	T(1) = B ;

	T.display() ;

	T.info() ;

	std::cout << std::endl << "Total CPU time = " << (dsecnd() - time_start ) << " seconds" << std::endl ;
  return 1;
}

