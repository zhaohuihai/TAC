/*
 * auxiliary.hpp
 *
 *  Created on: 2013-8-19
 *  Updated on: 2014-10-14
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

inline int mod(int x, int y) ;

template <typename C> std::vector<C> vec(const C& a0) ;
template <typename C> std::vector<C> vec(const C& a0, const C& a1) ;
template <typename C> std::vector<C> vec(const C& a0, const C& a1, const C& a2) ;
template <typename C> std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3) ;
template <typename C> std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4) ;
template <typename C> std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4, const C& a5) ;
template <typename C> std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4,
																				 const C& a5, const C& a6) ;
template <typename C> std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4,
																				 const C& a5, const C& a6, const C& a7) ;
template <typename C> std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4,
																				 const C& a5, const C& a6, const C& a7, const C& a8) ;
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

template <typename C>
std::vector<C> vec(const C& a0)
{
	std::vector<C> A ;
	A.push_back(a0) ;
	return A ;
}

template <typename C>
std::vector<C> vec(const C& a0, const C& a1)
{
	std::vector<C> A ;
	A.push_back(a0) ;
	A.push_back(a1) ;
	return A ;
}

template <typename C>
std::vector<C> vec(const C& a0, const C& a1, const C& a2)
{
	std::vector<C> A ;
	A.push_back(a0) ;
	A.push_back(a1) ;
	A.push_back(a2) ;
	return A ;
}

template <typename C>
std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3)
{
	std::vector<C> A ;
	A.push_back(a0) ;
	A.push_back(a1) ;
	A.push_back(a2) ;
	A.push_back(a3) ;
	return A ;
}

template <typename C>
std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4)
{
	std::vector<C> A ;
	A.push_back(a0) ;
	A.push_back(a1) ;
	A.push_back(a2) ;
	A.push_back(a3) ;
	A.push_back(a4) ;
	return A ;
}

template <typename C>
std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4, const C& a5)
{
	std::vector<C> A ;
	A.push_back(a0) ;
	A.push_back(a1) ;
	A.push_back(a2) ;
	A.push_back(a3) ;
	A.push_back(a4) ;
	A.push_back(a5) ;
	return A ;
}

template <typename C>
std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4,
									 const C& a5, const C& a6)
{
	std::vector<C> A ;
	A.push_back(a0) ;
	A.push_back(a1) ;
	A.push_back(a2) ;
	A.push_back(a3) ;
	A.push_back(a4) ;
	A.push_back(a5) ;
	A.push_back(a6) ;
	return A ;
}

template <typename C>
std::vector<C> vec(const C& a0, const C& a1, const C& a2, const C& a3, const C& a4,
									 const C& a5, const C& a6, const C& a7)
{
	std::vector<C> A ;
	A.push_back(a0) ;
	A.push_back(a1) ;
	A.push_back(a2) ;
	A.push_back(a3) ;
	A.push_back(a4) ;
	A.push_back(a5) ;
	A.push_back(a6) ;
	A.push_back(a7) ;
	return A ;
}

#endif /* AUXILIARY_HPP_ */
