/*
 * variation.h
 *
 *  Created on: 2011-6-27
 *      Author: zhaohuihai
 */

#ifndef VARIATION_H_
#define VARIATION_H_

#include "parameter.h"
#include "hamiltonian.h"
#include "wave_function.h"

class Variation
{
public:
	// constructor
	Variation(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave) ;
	// destructor
	~Variation() ;

	// ground state energy per site
	double energy ;

private:
	// apply variation and sweep all 9 blocks
	void applyVariation(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, int latticeSize) ;
	//==================================================================================================
	void optimize_A1(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, int latticeSize) ;

	Tensor compute_N1(Tensor &A1, ReducedTensor &rt) ;

	Tensor compute_H1(Hamiltonian &hamiltonian, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H1_12(Tensor &H12, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H1_14(Tensor &H14, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H1_23(Tensor &H23, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H1_47(Tensor &H47, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H1_36(Tensor &H36, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H1_78(Tensor &H78, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H1_69(Tensor &H69, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H1_89(Tensor &H89, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H1_25(Tensor &H25, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H1_45(Tensor &H45, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H1_56(Tensor &H56, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H1_58(Tensor &H58, WaveFunction &wave, ReducedTensor &rt) ;
	//--------------------------------------------------------------------------------------------------
	void optimize_A2(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, int latticeSize) ;

	Tensor compute_N2(Tensor &A2, ReducedTensor &rt) ;

	Tensor compute_H2(Hamiltonian &hamiltonian, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H2_12(Tensor &H12, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H2_23(Tensor &H23, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H2_25(Tensor &H25, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H2_14(Tensor &H14, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H2_36(Tensor &H36, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H2_47(Tensor &H47, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H2_69(Tensor &H69, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H2_78(Tensor &H78, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H2_89(Tensor &H89, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H2_45(Tensor &H45, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H2_56(Tensor &H56, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H2_58(Tensor &H58, WaveFunction &wave, ReducedTensor &rt) ;
	//---------------------------------------------------------------------------------------------------
	void optimize_A5(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, int latticeSize) ;

	Tensor compute_N5(Tensor &A5, ReducedTensor &rt) ;

	Tensor compute_H5(Hamiltonian &hamiltonian, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H5_12(Tensor &H12, WaveFunction &wave, ReducedTensor &rt) ;

	Tensor compute_H5_14(Tensor &H14, WaveFunction &wave, ReducedTensor &rt) ; // 12 -> 14
	Tensor compute_H5_36(Tensor &H36, WaveFunction &wave, ReducedTensor &rt) ; // 12 -> 14 -> 36
	Tensor compute_H5_78(Tensor &H78, WaveFunction &wave, ReducedTensor &rt) ; // 12 -> 14 -> 36 -> 78

	Tensor compute_H5_23(Tensor &H23, WaveFunction &wave, ReducedTensor &rt) ; // 12 -> 23
	Tensor compute_H5_47(Tensor &H47, WaveFunction &wave, ReducedTensor &rt) ; // 12 -> 23 -> 47
	Tensor compute_H5_69(Tensor &H69, WaveFunction &wave, ReducedTensor &rt) ; // 12 -> 23 -> 47 -> 69
	Tensor compute_H5_89(Tensor &H89, WaveFunction &wave, ReducedTensor &rt) ; // 12 -> 23 -> 47 -> 69 -> 89


	Tensor compute_H5_25(Tensor &H25, WaveFunction &wave, ReducedTensor &rt) ;
	Tensor compute_H5_45(Tensor &H45, WaveFunction &wave, ReducedTensor &rt) ; // 25 -> 45
	Tensor compute_H5_56(Tensor &H56, WaveFunction &wave, ReducedTensor &rt) ; // 25 -> 45 -> 56
	Tensor compute_H5_58(Tensor &H58, WaveFunction &wave, ReducedTensor &rt) ; // 25 -> 45 -> 56 -> 58

	//======================================================================================================
	// H*A = e*N*A
	void minimizeEnergyOneSite(Tensor &N, Tensor &H, Tensor &A, int latticeSize) ;
};

#endif /* VARIATION_H_ */
