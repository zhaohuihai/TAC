/*
 * Index.hpp
 *
 *  Created on: 2014Äê9ÔÂ26ÈÕ
 *      Author: zhaohuihai
 */

#ifndef INDEX_HPP_
#define INDEX_HPP_

#include <vector>

class Index
{
private:
	int _ind ;
	// count num of times every index used
	static std::vector<int> _inds_count ;

public:
	// constructor
	Index() ;
	Index(const Index& x1) ; // overload copy constructor
	// destructor
	~Index() ;

	int ind() const { return _ind ; }
	static std::vector<int> inds_count() { return _inds_count; }

	Index& operator = (const Index& x1) ; // copy Index
	bool operator == (const Index& x1) ;

};

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
}
// overload copy constructor
Index::Index(const Index& x1)
{
	_ind = x1.ind() ;
	_inds_count.at(_ind) ++ ;

//	std::cout << "index " << _ind << " is created for " << _inds_count[ _ind ] << " times"<< std::endl ;
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

#endif /* INDEX_HPP_ */
