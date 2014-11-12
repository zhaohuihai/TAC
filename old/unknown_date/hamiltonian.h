/*
 * hamiltonian.h
 *
 *  Created on: 2011-6-9
 *      Author: zhaohuihai
 */

#ifndef HAMILTONIAN_H_
#define HAMILTONIAN_H_

#include "tensor.h"

class Hamiltonian
{
public:
	Hamiltonian() ; // empty hamiltonian
	// initialize Hamiltonian of 4X4 lattice
	Hamiltonian(int d0) ;
	// initialize Hamiltonian of 3X3 lattice
	Hamiltonian(char debug) ;
	// destructor
	~Hamiltonian() ;
	// lattice size
	int length, width ;
	// physical dim of blocks
	int B1, B2, B3, B4, B6, B7, B8, B9 ;
	// 6 horizontal terms
	Tensor H12, H23, H45, H56, H78, H89 ;
	// 6 vertical terms
	Tensor H14, H47, H25, H58, H36, H69 ;
	// 8 terms inside blocks
	Tensor h1, h2, h3, h4, h6, h7, h8, h9 ;
	// 6 terms between every 2 horizontal blocks
	Tensor h12, h23, h45, h56, h78, h89 ;
	// 6 terms between every 2 vertical blocks
	Tensor h14, h47, h25, h58, h36, h69 ;
	// 5 terms of self interactions
	Tensor h22, h44, h55, h66, h88 ; // h55 would be never updated
	// ************* manipulation *****************
	// (S1,S2,S3,S4,S5,S6,S7,S8,S9) => (S3,S6,S9,S2,S5,S8,S1,S4,S7)
	void rotate() ; // rotate counter-clockwise

private:
	// AF Heisenberg model Hamiltonian of 2 sites h(m1',m2',m1,m2)= <m1',m2'|Q1*Q2|m1,m2>
	Tensor createHamiltonian_Heisenberg(int d0) ;
	// 6 horizontal terms
	inline void initialize_H12() ;
	//inline void initialize_H23(int d0) ;
	//inline void initialize_H45(int d0) ;
	//inline void initialize_H56(int d0) ;
	//inline void initialize_H78(int d0) ;
	inline void initialize_H89() ;
	// 6 vertical terms
	//inline void initialize_H14(int d0) ;
	inline void initialize_H47() ;
	//inline void initialize_H25(int d0) ;
	//inline void initialize_H58(int d0) ;
	inline void initialize_H36() ;
	//inline void initialize_H69(int d0) ;
	// 8 terms inside blocks
	inline void initialize_h1(int d0) ;
	inline void initialize_h2(int d0) ;
	inline void initialize_h3(int d0) ;
	inline void initialize_h4(int d0) ;
	inline void initialize_h6(int d0) ;
	inline void initialize_h7(int d0) ;
	inline void initialize_h8(int d0) ;
	inline void initialize_h9(int d0) ;
	// 6 terms between every 2 horizontal blocks
	inline void initialize_h12(int d0) ;
	inline void initialize_h23(int d0) ;
	inline void initialize_h45(int d0) ;
	inline void initialize_h56(int d0) ;
	inline void initialize_h78(int d0) ;
	inline void initialize_h89(int d0) ;
	// 6 terms between every 2 vertical blocks
	//inline void initialize_h14(int d0) ;
	//inline void initialize_h47(int d0) ;
	//inline void initialize_h25(int d0) ;
	//inline void initialize_h58(int d0) ;
	//inline void initialize_h36(int d0) ;
	//inline void initialize_h69(int d0) ;
	// 5 terms of self interactions
	//inline void initialize_h22(int d0) ;
	//inline void initialize_h44(int d0) ;
	//inline void initialize_h55(int d0) ;
	//inline void initialize_h66(int d0) ;
	//inline void initialize_h88(int d0) ;

	double delta(int a, int b) ;
};


#endif /* HAMILTONIAN_H_ */
