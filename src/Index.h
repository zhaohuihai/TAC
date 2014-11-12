/*
 * Index.h
 *
 *  Created on: 2014Äê10ÔÂ16ÈÕ
 *      Author: zhaohuihai
 */

#ifndef INDEX_H_
#define INDEX_H_

#include <vector>

class Index
{
private:
	int _ind ;
	int _dim ;
	// count num of times every index used
	static std::vector<int> _inds_count ;

public:
	// constructor
	Index() ;
	Index(const int d) ;  // initialize the dimension of the Index
	Index(const Index& x0) ; // overload copy constructor
	// combine the dim to create a new Index
	Index(const Index& x0, const Index& x1) ;
	Index(const Index& x0, const Index& x1, const Index& x2) ;
	Index(const Index& x0, const Index& x1, const Index& x2, const Index& x3) ;
	// destructor
	~Index() ;

	int ind() const { return _ind ; }
	int dim() const { return _dim ; }
	void put_dim(const int& d) { _dim = d ; }
	static std::vector<int> inds_count() { return _inds_count; }

	Index& operator = (const Index& x1) ; // copy Index
	bool operator == (const Index& x1) ;

};



#endif /* INDEX_H_ */
