/*
 * testTemplate.hpp
 *
 *  Created on: 2013��7��16��
 *      Author: ZhaoHuihai
 */

#ifndef TESTTEMPLATE_HPP_
#define TESTTEMPLATE_HPP_

#include "main.h"

template <class T> T findMin( T a, T b ) ;



//===================================================


template <class T> T findMin( T a, T b )
{
	return ( a < b ) ? a : b;
}



#endif /* TESTTEMPLATE_HPP_ */
