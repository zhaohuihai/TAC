// google c++ style: http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml

//#include <mpi.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>

#include <mkl.h>

//#include <boost/shared_ptr.hpp>

#include "TensorOperation.hpp"

#include "example.h"

int main()
{

//	double time_start = dsecnd() ;
	tensor_contraction() ;
	tensor_reshape() ;

	tensor_permute() ;

	tensor_svd() ;

//	std::cout << std::endl << "Total CPU time = " << (dsecnd() - time_start ) << " seconds" << std::endl ;
  return 1;
}

