/*
 * hamiltonian.cpp
 *
 *  Created on: 2011-6-9
 *      Author: zhaohuihai
 */
#include <iostream>
#include <valarray>
#include <cmath>

#include "hamiltonian.h"
#include "tensor.h"
#include "tensor_manipulation.h"

using namespace std ;
Hamiltonian::Hamiltonian() {}
Hamiltonian::Hamiltonian(int d0)
{
	// lattice size
	length = 4 ;
	width = 4 ;

	initialize_h12(d0) ;
	//* 8 terms inside blocks
	initialize_h1(d0) ;
	initialize_h2(d0) ;
	initialize_h3(d0) ;
	initialize_h4(d0) ;
	initialize_h6(d0) ;
	initialize_h7(d0) ;
	initialize_h8(d0) ;
	initialize_h9(d0) ;

	initialize_h23(d0) ;
	initialize_h45(d0) ;
	initialize_h56(d0) ;
	initialize_h78(d0) ;
	initialize_h89(d0) ;
	// Reflection symmetry
	h14 = h12 ;
	h47 = h23 ;
	h25 = h45 ;
	h58 = h56 ;
	h36 = h78 ;
	h69 = h89 ;
	// 5 terms of self interactions
	// h22(S2',new_S2',S2,new_S2) = <m2',new_m2'| Q2*new_Q2| m2, new_m2>
	h22 = h12 ;
	h44 = h12 ;
	h55 = h12 ;
	h66 = h78 ;
	h88 = h78 ;

	// 6 horizontal terms
	initialize_H12() ;
	H23 = h23 ;
	H45 = h45 ;
	H56 = h56 ;
	H78 = h78 ;
	initialize_H89() ;
	// 6 vertical terms
	H14 = h14 ;
	initialize_H47() ;
	H25 = h25 ;
	H58 = h58 ;
	initialize_H36() ;
	H69 = h69 ;

}
// initialize Hamiltonian of 3X3 lattice
Hamiltonian::Hamiltonian(char debug)
{
	if (debug == 'D')
	{
		cout << "initial lattice size: 3 X 3" << endl ;
	}
	else
	{
		cout << "Hamiltonian::Hamiltonian(char debug) error: unknown parameter" << endl ;
		exit(0) ;
	}

	// lattice size
	length = 3 ;
	width = 3 ;

	initialize_h12(2) ;

	B1 = 2 ;
	B2 = 2 ;
	B3 = 2 ;
	B4 = 2 ;
	B6 = 2 ;
	B7 = 2 ;
	B8 = 2 ;
	B9 = 2 ;

	h1 = Tensor(B1, B1) ;
	h2 = Tensor(B2, B2) ;
	h3 = Tensor(B3, B3) ;
	h4 = Tensor(B4, B4) ;
	h6 = Tensor(B6, B6) ;
	h7 = Tensor(B7, B7) ;
	h8 = Tensor(B8, B8) ;
	h9 = Tensor(B9, B9) ;

	h22 = h12 ;
	h44 = h12 ;
	h55 = h12 ;
	h66 = h12 ;
	h88 = h12 ;

	h23 = h12 ;
	h45 = h12 ;
	h56 = h12 ;
	h78 = h12 ;
	h89 = h12 ;
	h14 = h12 ;
	h47 = h12 ;
	h25 = h12 ;
	h58 = h12 ;
	h36 = h12 ;
	h69 = h12 ;

	H12 = h12 ;
	H23 = h12 ;
	H45 = h12 ;
	H56 = h12 ;
	H78 = h12 ;
	H89 = h12 ;
	H14 = h12 ;
	H47 = h12 ;
	H25 = h12 ;
	H58 = h12 ;
	H36 = h12 ;
	H69 = h12 ;

}
//========================================================================================================
//* AF Heisenberg model Hamiltonian of 2 sites h(m1',m2',m1,m2)= <m1',m2'|Q1*Q2|m1,m2>
Tensor Hamiltonian::createHamiltonian_Heisenberg(int d0)
{
	//* total spin
	double S = ((double)d0 - 1) / 2 ;

	int a [] = {d0, d0, d0, d0} ;
	valarray<int> dim(a, 4) ;
	Tensor h = Tensor(dim) ;

	double q1, q2 ;
	int m1, m2 ;
	//* assign values to diagonal entries
	for (m2 = 1; m2 <= d0; m2 ++)
	{
		q2 = S - (double)m2 + 1 ; // quantum number
		for (m1 = 1; m1 <= d0; m1 ++)
		{
			q1 = S - (double)m1 + 1 ;
			h(m1, m2, m1, m2) = h(m1, m2, m1, m2) + q1 * q2 ;
		}
	}
	//******************************************************************
	// XXZ model
	//h = h * 4.0 ;
	//*********************************************************************
	//* assign values to off-diagonal entries
	int m1p, m2p ;
	double SpSm, SmSp ;
	for (m2 = 1 ; m2 <= d0; m2 ++)
	{
		q2 = S - (double)m2 + 1 ;
		for (m1 = 1; m1 <= d0; m1 ++)
		{
			q1 = S - (double)m1 + 1 ;
			// SpSm = (1/2)((S1+)(S2-))
			SpSm = 0.5 * sqrt((S - q1) * (S + q1 + 1) * (S + q2) * (S - q2 + 1)) ;
			if (SpSm != 0) // <(m1-1),(m2+1)|SpSm|m1,m2>
			{
				m1p = m1 - 1 ;
				m2p = m2 + 1 ;
				h(m1p, m2p, m1, m2) = h(m1p, m2p, m1, m2) + SpSm ;
			}
			// SmSp = (1/2)((S1-)(S2+))
			SmSp = 0.5 * sqrt((S + q1) * (S - q1 + 1) * (S - q2) * (S + q2 + 1)) ;
			if (SmSp != 0) // <(m1+1),(m2-1)|SpSm|m1,m2>
			{
				m1p = m1 + 1 ;
				m2p = m2 - 1 ;
				h(m1p, m2p, m1, m2) = h(m1p, m2p, m1, m2) + SmSp ;
			}
		}
	}
	//*************************************
	//h = 0.0 - h ; // FM
	//*************************************
	return h ;
}

//* h1(S1', S1) = 0
inline void Hamiltonian::initialize_h1(int d0)
{
	B1 = d0 ;
	int a [] = {B1, B1} ;
	valarray<int> dim(a, 2) ;
	h1 = Tensor(dim) ;
}
// h2(S2', S2) = 0
inline void Hamiltonian::initialize_h2(int d0)
{
	B2 = d0 ;
	h2 = h1 ;
}
// h3(S3',S3) = h3(m3', m4'; m3, m4) = <m3',m4'|Q3*Q4|m3, m4>
inline void Hamiltonian::initialize_h3(int d0)
{
	B3 = d0 * d0 ;
	h3 = h12 ;
	valarray<int> h3_dim (2) ;
	h3_dim[0] = h12.dim[0] * h12.dim[1] ;
	h3_dim[1] = h12.dim[2] * h12.dim[3] ;
	//reshape
	//h3.dim = h3_dim ;
	h3.reshape(h3_dim) ;
	//printMatrix(h3) ;
}
// h4(S4',S4) = 0
inline void Hamiltonian::initialize_h4(int d0)
{
	B4 = d0 ;
	h4 = h1 ;
}
// h6(S6',S6)
inline void Hamiltonian::initialize_h6(int d0)
{
	B6 = B3 ;
	h6 = h3 ;
}
// h7(S7',S7)
inline void Hamiltonian::initialize_h7(int d0)
{
	B7 = B3 ;
	h7 = h3 ;
}
// h8(S8',S8)
inline void Hamiltonian::initialize_h8(int d0)
{
	B8 = B3 ;
	h8 = h3 ;
}
// h9(S9',S9) = h9(m13',m14',m15',m16';m13,m14,m15,m16) =
// <m13',m14',m15',m16'|[Q13*Q14 + Q13*Q15 + Q14*Q16 + Q15*Q16]|m13,m14,m15,m16>
inline void Hamiltonian::initialize_h9(int d0)
{
	//B9 = pow(d0, 4) ;
	B9 = d0 * d0 * d0 * d0 ;
	valarray<int> dim(d0, 8) ;
	h9 = Tensor(dim) ;
	int m13, m14, m15, m16 ;
	int m13p, m14p, m15p, m16p ;
	for (m13 = 1 ; m13 <= d0; m13 ++)
	{
		for (m14 = 1 ; m14 <= d0; m14 ++)
		{
			for (m15 = 1; m15 <= d0; m15 ++)
			{
				for (m16 = 1 ; m16 <= d0; m16 ++)
				{
					for (m13p = 1 ; m13p <= d0; m13p ++)
					{
						for (m14p = 1 ; m14p <= d0; m14p ++)
						{
							for (m15p = 1 ; m15p <= d0; m15p ++)
							{
								for (m16p = 1 ; m16p <= d0; m16p ++)
								{
									h9(m13p, m14p, m15p, m16p, m13, m14, m15, m16) =
											h12(m13p, m14p, m13, m14) * delta(m15p, m15) * delta(m16p, m16) +
											h12(m13p, m15p, m13, m15) * delta(m14p, m14) * delta(m16p, m16) +
											h12(m14p, m16p, m14, m16) * delta(m13p, m13) * delta(m15p, m15) +
											h12(m15p, m16p, m15, m16) * delta(m13p, m13) * delta(m14p, m14) ;
								}
							}
						}
					}
				}
			}
		}
	}
	valarray<int> h9_dim(B9, 2) ;
	// reshape
	h9.dim = h9_dim ;

}

// h12(m1',m2',m1,m2) = <m1',m2'|S1*S2|m1,m2>
// ~~ h12(S1',S2',S1,S2)
inline void Hamiltonian::initialize_h12(int d0)
{
	h12 = createHamiltonian_Heisenberg(d0) ;
}
//h23(S2',S3',S2,S3) = h23(m2',m3',m4',m2,m3,m4) =
// <m2',m3',m4'|Q2*Q3|m2,m3,m4>
inline void Hamiltonian::initialize_h23(int d0)
{
	valarray<int> dim(d0, 6) ;
	h23 = Tensor(dim) ;
	int m2, m3, m4 ;
	int m2p, m3p, m4p ;
	for (m2 = 1 ; m2 <= d0; m2 ++)
	{
		for (m3 = 1 ; m3 <= d0; m3 ++)
		{
			for (m4 = 1 ; m4 <= d0; m4 ++)
			{
				for (m2p = 1 ; m2p <= d0; m2p ++)
				{
					for (m3p = 1 ; m3p <= d0; m3p ++)
					{
						for (m4p = 1 ; m4p <= d0; m4p ++)
						{
							h23(m2p, m3p, m4p, m2, m3, m4) = h12(m2p, m3p, m2, m3) * delta(m4p, m4) ;
						}
					}
				}

			}
		}
	}
	valarray<int> h23_dim(4) ;
	h23_dim[0] = B2 ;
	h23_dim[1] = B3 ;
	h23_dim[2] = B2 ;
	h23_dim[3] = B3 ;
	// reshape (m2',m3',m4',m2,m3,m4) => (m2',(m3',m4'),m2,(m3,m4))
	// ~~ (S2',S3',S2,S3)
	h23.dim = h23_dim ;
}
// h45(m5',m6',m5,m6) = <m5',m6'|Q5*Q6|m5,m6>
// ~~ h45(S4',S5',S4,S5)
inline void Hamiltonian::initialize_h45(int d0)
{
	h45 = h12 ;
}
inline void Hamiltonian::initialize_h56(int d0)
{
	h56 = h23 ;
}
// h78(S7',S8',S7,S8) = h78(m9',m10',m11',m12',m9,m10,m11,m12) =
// <m9',m10',m11',m12'| Q9*Q11 + Q10*Q12 | m9,m10,m11,m12>
inline void Hamiltonian::initialize_h78(int d0)
{
	valarray<int> dim(d0, 8) ;
	h78 = Tensor(dim) ;
	int m9, m10, m11, m12 ;
	int m9p, m10p, m11p, m12p ;
	for (m9 = 1 ; m9 <= d0; m9 ++)
	{
		for (m10 = 1 ; m10 <= d0; m10 ++)
		{
			for (m11 = 1 ; m11 <= d0; m11 ++)
			{
				for (m12 = 1 ; m12 <= d0; m12 ++)
				{
					for (m9p = 1 ; m9p <= d0; m9p ++)
					{
						for (m10p = 1 ; m10p <= d0; m10p ++)
						{
							for (m11p = 1 ; m11p <= d0; m11p ++)
							{
								for (m12p = 1 ; m12p <= d0; m12p ++)
								{
									h78(m9p, m10p, m11p, m12p, m9, m10, m11, m12) =
											h12(m9p, m11p, m9, m11) * delta(m10p, m10) * delta(m12p, m12) +
											h12(m10p, m12p, m10, m12) * delta(m9p, m9) * delta(m11p, m11) ;
								}
							}
						}
					}
				}
			}
		}
	}
	valarray<int> h78_dim(4) ;
	h78_dim[0] = B7 ;
	h78_dim[1] = B8 ;
	h78_dim[2] = B7 ;
	h78_dim[3] = B8 ;
	// reshape
	// h78.dim = h78_dim ;
	h78.reshape(B7, B8, B7, B8) ;
}
// h89(S8',S9',S8,S9) = h89(m11',m12',m13',m14',m15',m16',m11,m12,m13,m14,m15,m16) =
// <m11',m12',m13',m14',m15',m16'| Q11*Q13 + Q12 * Q15 | m11,m12,m13,m14,m15,m16>
inline void Hamiltonian::initialize_h89(int d0)
{
	valarray<int> dim(d0, 12) ;
	h89 = Tensor(dim) ;
	int m11, m12, m13, m14, m15, m16 ;
	int m11p, m12p, m13p, m14p, m15p, m16p ;
	for (m11 = 1 ; m11 <= d0; m11 ++)
	{
		for (m12 = 1 ; m12 <= d0; m12 ++)
		{
			for (m13 = 1 ; m13 <= d0; m13 ++)
			{
				for (m14 = 1 ; m14 <= d0; m14 ++)
				{
					for (m15 = 1 ; m15 <= d0; m15 ++)
					{
						for (m16 = 1 ; m16 <= d0; m16 ++)
						{
							for (m11p = 1 ; m11p <= d0; m11p ++)
							{
								for (m12p = 1 ; m12p <= d0; m12p ++)
								{
									for (m13p = 1 ; m13p <= d0; m13p ++)
									{
										for (m14p = 1 ; m14p <= d0; m14p ++)
										{
											for (m15p = 1 ; m15p <= d0; m15p ++)
											{
												for (m16p = 1 ; m16p <= d0; m16p ++)
												{
													h89(m11p, m12p, m13p, m14p, m15p, m16p, m11, m12, m13, m14, m15, m16) =
															h12(m11p, m13p, m11, m13) * delta(m12p, m12) * delta(m14p, m14) * delta(m15p, m15) * delta(m16p, m16) +
															h12(m12p, m15p, m12, m15) * delta(m11p, m11) * delta(m13p, m13) * delta(m14p, m14) * delta(m16p, m16) ;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	valarray<int> h89_dim(4) ;
	h89_dim[0] = B8 ;
	h89_dim[1] = B9 ;
	h89_dim[2] = B8 ;
	h89_dim[3] = B9 ;
	// reshape
	h89.dim = h89_dim ;
}
// H12(S1',S2',S1,S2) = h1(S1', S1) + h2(S2', S2) + h12(S1',S2',S1,S2)
inline void Hamiltonian::initialize_H12()
{
	int a[] = {B1, B2, B1, B2} ;
	valarray<int> dim(a, 4) ;
	H12 = Tensor(dim) ;
	int S1, S2 ;
	int S1p, S2p ;
	for (S1 = 1 ; S1 <= B1; S1 ++)
	{
		for (S2 = 1 ; S2 <= B2; S2 ++)
		{
			for (S1p = 1 ; S1p <= B1; S1p ++)
			{
				for (S2p = 1 ; S2p <= B2; S2p ++)
				{
					H12(S1p, S2p, S1, S2) = h1(S1p,S1) * delta(S2p, S2) + h2(S2p, S2) * delta(S1p, S1) + h12(S1p, S2p, S1, S2) ;
				}
			}
		}
	}
}
// H89(S8',S9',S8,S9) = h8(S8',S8) + h9(S9',S9) + h89(S8',S9',S8,S9)
inline void Hamiltonian::initialize_H89()
{
	int a[] = {B8, B9, B8, B9} ;
	valarray<int> dim(a, 4) ;
	H89 = Tensor(dim) ;
	int S8, S9 ;
	int S8p, S9p ;
	for (S8 = 1 ; S8 <= B8; S8 ++)
	{
		for (S9 = 1 ; S9 <= B9; S9 ++)
		{
			for (S8p = 1 ; S8p <= B8; S8p ++)
			{
				for (S9p = 1 ; S9p <= B9; S9p ++)
				{
					H89(S8p, S9p, S8, S9) = h8(S8p, S8) * delta(S9p, S9) + h9(S9p, S9) * delta(S8p, S8) + h89(S8p, S9p, S8, S9) ;
				}
			}
		}
	}
}
// H47(S4',S7',S4,S7) = h4(S4', S4) + h7(S7',S7) + h47(S4',S7',S4,S7)
inline void Hamiltonian::initialize_H47()
{
	int a[] = {B4, B7, B4, B7} ;
	valarray<int> dim(a, 4) ;
	H47 = Tensor(dim) ;
	int S4, S7 ;
	int S4p, S7p ;
	for (S4 = 1 ; S4 <= B4; S4 ++)
	{
		for (S7 = 1 ; S7 <= B7; S7 ++)
		{
			for (S4p = 1 ; S4p <= B4; S4p ++)
			{
				for (S7p = 1 ; S7p <= B7; S7p ++)
				{
					H47(S4p, S7p, S4, S7) = h4(S4p, S4) * delta(S7p, S7) + h7(S7p, S7) * delta(S4p, S4) + h47(S4p, S7p, S4, S7) ;
				}
			}
		}
	}
}
// H36(S3',S6',S3,S6) = h3(S3',S3) + h6(S6',S6) + h36(S3',S6',S3,S6)
inline void Hamiltonian::initialize_H36()
{
	int a[] = {B3, B6, B3, B6} ;
	valarray<int> dim(a, 4) ;
	H36 = Tensor(dim) ;
	int S3, S6 ;
	int S3p, S6p ;
	for (S3 = 1 ; S3 <= B3; S3 ++)
	{
		for (S6 = 1 ; S6 <= B6; S6 ++)
		{
			for (S3p = 1 ; S3p <= B3; S3p ++)
			{
				for (S6p = 1 ; S6p <= B6; S6p ++)
				{
					H36(S3p, S6p, S3, S6) = h3(S3p,S3) * delta(S6p, S6) + h6(S6p, S6) * delta(S3p, S3) + h36(S3p, S6p, S3, S6) ;
				}
			}
		}
	}
}

double Hamiltonian::delta(int a, int b)
{
	if (a == b)
	{
		return 1.0 ;
	}
	else
	{
		return 0.0 ;
	}
}
// ************* manipulation *****************
void Hamiltonian::rotate()
{
	Hamiltonian hamiltonian0 = *this ;
	length = hamiltonian0.width ;
	width = hamiltonian0.length ;
	B1 = hamiltonian0.B3 ;
	B2 = hamiltonian0.B6 ;
	B3 = hamiltonian0.B9 ;
	B4 = hamiltonian0.B2 ;
	B6 = hamiltonian0.B8 ;
	B7 = hamiltonian0.B1 ;
	B8 = hamiltonian0.B4 ;
	B9 = hamiltonian0.B7 ;

	int a [] = {2, 1, 4, 3} ;
	valarray<int> order(a, 4) ;

	H12 = hamiltonian0.H36 ;
	H23 = hamiltonian0.H69 ;
	H45 = hamiltonian0.H25 ;
	H56 = hamiltonian0.H58 ;
	H78 = hamiltonian0.H14 ;
	H89 = hamiltonian0.H47 ;
	// H14(S4',S1',S4,S1)
	H14 = hamiltonian0.H23 ;
	// H14(S4',S1',S4,S1) => H14(S1',S4',S1,S4)
	H14.permute(order) ; // order = [2, 1, 4, 3]

	// H47(S7',S4',S7,S4)
	H47 = hamiltonian0.H12 ;
	// H47(S7',S4',S7,S4) => H47(S4',S7',S4,S7)
	H47.permute(order) ;

	H25 = hamiltonian0.H56 ;
	H25.permute(order) ;

	H58 = hamiltonian0.H45 ;
	H58.permute(order) ;

	H36 = hamiltonian0.H89 ;
	H36.permute(order) ;

	H69 = hamiltonian0.H78 ;
	//cout << H69.dim.size() << endl ;
	H69.permute(order) ;
	//exit(0) ;
	h1 = hamiltonian0.h3 ;
	h2 = hamiltonian0.h6 ;
	h3 = hamiltonian0.h9 ;
	h4 = hamiltonian0.h2 ;
	h6 = hamiltonian0.h8 ;
	h7 = hamiltonian0.h1 ;
	h8 = hamiltonian0.h4 ;
	h9 = hamiltonian0.h7 ;

	h12 = hamiltonian0.h36 ;
	h23 = hamiltonian0.h69 ;
	h45 = hamiltonian0.h25 ;
	h56 = hamiltonian0.h58 ;
	h78 = hamiltonian0.h14 ;
	h89 = hamiltonian0.h47 ;
	// h14(S4',S1',S4,S1)
	h14 = hamiltonian0.h23 ;
	// h14(S4',S1',S4,S1) => h14(S1',S4',S1,S4)
	h14.permute(order) ; // order = [2, 1, 4, 3]
	// h47(S7',S4',S7,S4)
	h47 = hamiltonian0.h12 ;
	// h47(S7',S4',S7,S4) => h47(S4',S7',S4,S7)
	h47.permute(order) ;
	h25 = hamiltonian0.h56 ;
	h25.permute(order) ;
	h58 = hamiltonian0.h45 ;
	h58.permute(order) ;
	h36 = hamiltonian0.h89 ;
	h36.permute(order) ;
	h69 = hamiltonian0.h78 ;
	h69.permute(order) ;

	h22 = hamiltonian0.h66 ;
	h44 = hamiltonian0.h22 ;
	h66 = hamiltonian0.h88 ;
	h88 = hamiltonian0.h44 ;
}
//destructor
Hamiltonian::~Hamiltonian() {}
