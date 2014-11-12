/*
 * auxiliary.hpp
 *
 *  Created on: 2013年8月19日
 *      Author: ZhaoHuihai
 */

#ifndef AUXILIARY_HPP_
#define AUXILIARY_HPP_

// C++ Standard Template Library
//	1. C library
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cassert>
//	2. Containers
#include <vector>
//	3. Input/Output
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
//	4. Other
#include <algorithm>
#include <string>
#include <complex>
#include <utility>
//=============================================================

inline void makeDir(std::string dirName) ;

template <typename C> std::string num2str(C a) ;
template <typename C> std::string num2str(C a, int precision) ;

template <typename C> void disp(C a) ;

int mod(int x, int y) ;
//===========================================================

void makeDir(std::string dirName)
{
	std::string makeDirectory = "mkdir -p " ;
	makeDirectory = makeDirectory + dirName ;
	system( &(makeDirectory[0]) ) ;
}

template <typename C>
std::string num2str(C a)
{
	std::ostringstream ss ;
	ss << a ;
	return ss.str() ;

}

template <typename C>
std::string num2str(C a, int precision)
{
	std::ostringstream ss ;
	ss << std::setprecision(precision) << a ;
	return ss.str() ;
}

int mod(int x, int y)
{
	int r = x % y ;
	if (r < 0)
	{
		r = r + y ;
	}
	return r ;
}

#endif /* AUXILIARY_HPP_ */
