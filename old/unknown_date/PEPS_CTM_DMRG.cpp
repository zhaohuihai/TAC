/*
 * PEPS_CTM_DMRG.cpp
 *
 *  Created on: 2011-6-27
 *      Author: zhaohuihai
 */

#include "main.h"

using namespace std ;

PEPS_CTM_DMRG::PEPS_CTM_DMRG(Parameter &parameter)
{
	//Hamiltonian hamiltonian = Hamiltonian(parameter.d0) ;
	Hamiltonian hamiltonian = Hamiltonian('D') ;  // 3X3
	WaveFunction wave = WaveFunction(parameter, hamiltonian) ;
	// determine the ground state wave function by variational method
	Variation groundState = Variation(parameter, hamiltonian, wave) ;
	DMRG dmrg ;
	while ( (hamiltonian.length < parameter.lengthFinal) || (hamiltonian.width < parameter.widthFinal) )
	{
		if (hamiltonian.length >= parameter.lengthFinal)
		{
			// rotate wave and hamiltonian
			parameter.rotate() ;
			hamiltonian.rotate() ;
			wave.rotate() ;
		}
		// block 1,4,7 are updated. Lattice length is increased by 1.
		dmrg = DMRG(parameter, hamiltonian, wave) ;
		// rotate wave and hamiltonian
		parameter.rotate() ;
		hamiltonian.rotate() ;
		wave.rotate() ;

		cout << "length = " << hamiltonian.length << ", width = " << hamiltonian.width << endl ;

		wave = WaveFunction(parameter, hamiltonian) ;
		groundState = Variation(parameter, hamiltonian, wave) ;
		//cout << "energy = " << groundState.energy << endl ;
	}
}

// =========================================destructor==============================================
PEPS_CTM_DMRG::~PEPS_CTM_DMRG() {}
