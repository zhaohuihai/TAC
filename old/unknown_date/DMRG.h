/*
 * DMRG.h
 *
 *  Created on: 2011-6-27
 *      Author: zhaohuihai
 */

#ifndef DMRG_H_
#define DMRG_H_

class DMRG
{
public:
	// constructor
	DMRG() ;
	DMRG(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave) ;
	// destructor
	~DMRG() ;

private:
	void updateHamiltonian_12(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, Tensor &U12) ;
	Tensor computeDensMat_12(WaveFunction &wave) ;
	Tensor update_h1(Tensor H12, Tensor U12) ;
	Tensor update_h12(Tensor h22, Tensor U12) ;
	Tensor update_H12(Tensor h1, Tensor h2, Tensor h12) ;
	//-----------------------------------------------------------------------------------------------------------------------------------
	void updateHamiltonian_78(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, Tensor &U78) ;
	Tensor computeDensMat_78(WaveFunction &wave) ;
	Tensor update_h7(Tensor h7, Tensor h8, Tensor h78, Tensor U78) ;
	//-----------------------------------------------------------------------------------------------------------------------------------
	void updateHamiltonian_45(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, Tensor &U12, Tensor &U78) ;
	Tensor computeDensMat_45(WaveFunction &wave) ;
	Tensor update_h4(Tensor h4, Tensor h45, Tensor U45) ;
	Tensor update_h14(Tensor h14, Tensor h25, Tensor U12, Tensor U45) ;
	//Tensor update_h47(Tensor h47, Tensor h58, Tensor U45, Tensor U78) ;
	//Tensor update_H47(Tensor h4, Tensor h7, Tensor h47) ;
	Tensor update_h44(Tensor h44, Tensor h55, Tensor U45) ;

	//------------------------------------------------------------------------------------------------------------------------------------
	double truncateDensMat(int dSave, Tensor &U, Tensor &Lambda) ;

	double delta(int a, int b) ;

};

#endif /* DMRG_H_ */
