/*
 * example.cpp
 *
 *  Created on: 2014��10��16��
 *      Author: zhaohuihai
 */

#include "example.h"

Index test(Index& x1)
{
	return x1 ;

}

void test2(const Index& y)
{
	std::cout << " ind " << y.ind() << std::endl ;
	std::cout << " dim " << y.dim() << std::endl ;
}
