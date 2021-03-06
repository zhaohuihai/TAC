/*
 * Index.cpp
 *
 *  Created on: 2014-10-16
 *  Updated on: 2014-11-04
 *      Author: zhaohuihai
 */

#include <stdlib.h>
#include <iostream>

#include "Index.h"

std::vector<int> Index::_inds_count = std::vector<int>(0) ;

//=========================================================
// constructor
Index::Index()
{
	bool ind_found = false ;
	for (int i = 0 ; i < _inds_count.size(); i ++)
	{
		if (_inds_count[i] == 0) // i-th index is unused
		{
			_ind = i ;
			_inds_count[i] = 1 ;
			ind_found = true ;
			break ;
		}
	}

	if ( !ind_found ) // add one new index
	{
		_ind = _inds_count.size() ;
		_inds_count.push_back(1) ;
	}

//	std::cout << "index " << _ind << " is created." << std::endl ;

	_dim = -1 ; // Index dimension is not defined.
}

Index::Index(const int d)
{
	bool ind_found = false ;
	for (int i = 0 ; i < _inds_count.size(); i ++)
	{
		if (_inds_count[i] == 0) // i-th index is unused
		{
			_ind = i ;
			_inds_count[i] = 1 ;
			ind_found = true ;
			break ;
		}
	}

	if ( !ind_found ) // add one new index
	{
		_ind = _inds_count.size() ;
		_inds_count.push_back(1) ;
	}

//	std::cout << "index " << _ind << " is created." << std::endl ;

	_dim = d ; // Index dimension is defined.
}

// overload copy constructor
Index::Index(const Index& x0)
{
	_ind = x0.ind() ;
	_inds_count.at(_ind) ++ ;

	_dim = x0.dim() ;
//	std::cout << "index " << _ind << " is created for " << _inds_count[ _ind ] << " times"<< std::endl ;
}

// combine the dim to create a new Index
Index::Index(const Index& x0, const Index& x1)
{

//	*this = Index() ;
	bool ind_found = false ;
	for (int i = 0 ; i < _inds_count.size(); i ++)
	{
		if (_inds_count[i] == 0) // i-th index is unused
		{
			_ind = i ;
			_inds_count[i] = 1 ;
			ind_found = true ;
			break ;
		}
	}

	if ( !ind_found ) // add one new index
	{
		_ind = _inds_count.size() ;
		_inds_count.push_back(1) ;
	}
	//------------------------------------------
	if ( (x0.dim() >= 0 ) && ( x1.dim() >= 0 ) )
	{
		_dim =  x0.dim() * x1.dim() ;
	}
	else
	{
		_dim = - 1 ;
	}
}

Index::Index(const Index& x0, const Index& x1, const Index& x2)
{
	bool ind_found = false ;
	for (int i = 0 ; i < _inds_count.size(); i ++)
	{
		if (_inds_count[i] == 0) // i-th index is unused
		{
			_ind = i ;
			_inds_count[i] = 1 ;
			ind_found = true ;
			break ;
		}
	}

	if ( !ind_found ) // add one new index
	{
		_ind = _inds_count.size() ;
		_inds_count.push_back(1) ;
	}
	//------------------------------------------
	if ( (x0.dim() >= 0 ) && ( x1.dim() >= 0 ) && (x2.dim() >= 0) )
	{
		_dim =  x0.dim() * x1.dim() * x2.dim() ;
	}
	else
	{
		_dim = - 1 ;
	}
}
Index::Index(const Index& x0, const Index& x1, const Index& x2, const Index& x3)
{
	bool ind_found = false ;
	for (int i = 0 ; i < _inds_count.size(); i ++)
	{
		if (_inds_count[i] == 0) // i-th index is unused
		{
			_ind = i ;
			_inds_count[i] = 1 ;
			ind_found = true ;
			break ;
		}
	}

	if ( !ind_found ) // add one new index
	{
		_ind = _inds_count.size() ;
		_inds_count.push_back(1) ;
	}
	//------------------------------------------
	if ( (x0.dim() >= 0 ) && ( x1.dim() >= 0 ) && (x2.dim() >= 0) && (x3.dim() >= 0) )
	{
		_dim =  x0.dim() * x1.dim() * x2.dim() * x3.dim() ;
	}
	else
	{
		_dim = - 1 ;
	}
}

// destructor
Index::~Index()
{
	_inds_count.at(_ind) -- ;

//	std::cout << "index " << _ind << " is destroyed" << std::endl ;

	if ( _inds_count.at(_ind) < 0 )
	{
		std::cout << "Inexistent Index is destroyed" << std::endl ;
		exit(0) ;
	}
}

// overload =
Index& Index::operator =(const Index& x1)
{
	_inds_count.at(_ind) -- ;
	_ind = x1.ind() ;
	_inds_count.at(_ind) ++ ;

	_dim = x1.dim() ;

	return *this ;
}

// overload ==
bool Index::operator ==(const Index& x1)
{
	if (_ind == x1.ind())
	{
		return true ;
	}
	else
	{
		return false ;
	}
}
