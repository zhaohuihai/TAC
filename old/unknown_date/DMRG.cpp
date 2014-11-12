/*
 * DMRG.cpp
 *
 *  Created on: Nov 29, 2011
 *      Author: zhaohuihai
 */

#include "main.h"

using namespace std ;
// constructor
DMRG::DMRG() { }

DMRG::DMRG(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave)
{

	Tensor U12 ;
	updateHamiltonian_12(parameter, hamiltonian, wave, U12) ;
	//cout << "B1 = " << hamiltonian.B1 << endl ;

	Tensor U78 ;
	updateHamiltonian_78(parameter, hamiltonian, wave, U78) ;
	//cout << "B7 = " << hamiltonian.B7 << endl ;

	updateHamiltonian_45(parameter, hamiltonian, wave, U12, U78) ;
	//cout << "B4 = " << hamiltonian.B4 << endl ;

	hamiltonian.length = hamiltonian.length + 1 ;
}

//========================================================member function============================================
void DMRG::updateHamiltonian_12(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, Tensor &U12)
{
	// compute reduced density matrix of block 1,2
	Tensor densMat = computeDensMat_12(wave) ; // densMat(S1,S2,S1',S2')

	int dS1 = densMat.dim[0] ;
	int dS2 = densMat.dim[1] ;
	// densMat(S1,S2,S1',S2') => densMat((S1,S2),(S1',S2'))
	densMat.reshape(dS1 * dS2, dS1 * dS2) ;

	U12 = Tensor(dS1 * dS2, dS1 * dS2) ;
	// densMat((S1,S2),(S1',S2')) = sum{S1_new}_[U12((S1,S2),S1_new)*Lambda(S1_new)*U12((S1',S2'),S1_new)]
	Tensor Lambda = symEig(densMat, U12) ;

	double truncErr = truncateDensMat(parameter.d1, U12, Lambda) ;
	if (parameter.disp == 1)
	{
		cout << "truncation error 12: " << truncErr << endl ;
	}

	// U12((S1,S2),S1_new)
	int dS1_new = U12.dim[1] ;
	// U12((S1,S2),S1_new) => U12(S1,S2,S1_new)
	U12.reshape(dS1, dS2, dS1_new) ;

	hamiltonian.B1 = Lambda.dim[0] ;

	// h1_new(S1_new, S1_new') = sum{S1,S2,S1',S2'}_[U12(S1,S2,S1_new)*H12(S1,S2,S1',S2')*U12(S1',S2',S1_new')]
	hamiltonian.h1 = update_h1(hamiltonian.H12, U12) ;

	// h2_new = h2

	// h12_new(S1_new,S2_new,S1_new',S2_new')
	// = sum{S1,S2,S1',S2'}_[U12(S1,S2,S1_new)*h22(S2,S2_new,S2',S2_new')*U12(S1',S2',S1_new')*delta(S1,S1')]
	// = sum{S1,S2,S2'}_[U12(S1,S2,S1_new)*h22(S2,S2_new,S2',S2_new')*U12(S1,S2',S1_new')]
	hamiltonian.h12 = update_h12(hamiltonian.h22, U12) ;

	// H12_new = h1_new + h2_new + h12_new
	hamiltonian.H12 = update_H12(hamiltonian.h1, hamiltonian.h2, hamiltonian.h12) ;
}
// densMat(S1,S2,S1',S2')
Tensor DMRG::computeDensMat_12(WaveFunction &wave)
{
	ReducedTensor rt = ReducedTensor(wave) ;

	// 1: T8*T9
	// densMat(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor densMat = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*densMat
	// densMat(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*densMat(X5,Y4,Y6)]
	densMat = contractTensors(rt.T7, 1, densMat, 1) ;

	// 3: T6*densMat
	// densMat(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*densMat(Y2,Y4,Y6)]
	densMat = contractTensors(rt.T6, 3, densMat, 3) ;

	// 4: T5*densMat
	// densMat(X3,Y3,Y5,Y2) = sum{X4, Y4}_[T5(X3,X4,Y3,Y4)*densMat(X4,Y5,Y2,Y4)]
	densMat = contractTensors(rt.T5, 2, 4, densMat, 1, 4) ;

	// 5: T4*densMat
	// densMat(Y1,Y3,Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*densMat(X3,Y3,Y5,Y2)]
	densMat = contractTensors(rt.T4, 1, 3, densMat, 1, 4) ;

	// 6: T3*densMat
	// densMat(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*densMat(Y1,Y3,Y5)]
	densMat = contractTensors(rt.T3, 2, densMat, 3) ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	// A4(x3,y1,y2,S4)
	int dx3 = wave.A4.dim[0] ;
	int dy1 = wave.A4.dim[1] ;
	int dy2 = wave.A4.dim[2] ;

	// densMat(X2,Y1,Y3) => densMat(x2,x2',y1,y1',y3,y3')
	densMat.reshape(dx2, dx2, dy1, dy1, dy3, dy3) ;

	// 7-1: A1*A2
	// AA(y1,S1,x2,y3,S2) = sum{x1}_[A1(x1,y1,S1)*A2(x1,x2,y3,S2)]
	Tensor AA = contractTensors(wave.A1, 1, wave.A2, 1) ;

	// 7-2: AA'*densMat
	// densMat(S1',S2',x2,y1,y3) = sum{y1',x2',y3'}_[AA(y1',S1',x2',y3',S2')*densMat(x2,x2',y1,y1',y3,y3')]
	densMat = contractTensors(AA, 1, 3, 4, densMat, 4, 2, 6) ;

	// 7-3: AA*densMat
	// densMat(S1,S2,S1',S2') = sum{y1,x2,y3}_[AA(y1,S1,x2,y3,S2)*densMat(S1',S2',x2,y1,y3)]
	densMat = contractTensors(AA, 1, 3, 4, densMat, 4, 3, 5) ;

	return densMat ;
}

// h1(S1_new, S1_new') = sum{S1,S2,S1',S2'}_[U12(S1,S2,S1_new)*H12(S1,S2,S1',S2')*U12(S1',S2',S1_new')]
Tensor DMRG::update_h1(Tensor H12, Tensor U12)
{
	// 1: H12*U12
	// h1(S1,S2,S1_new') = sum{S1',S2'}_[H12(S1,S2,S1',S2')*U12(S1',S2',S1_new')]
	Tensor h1 = contractTensors(H12, 3, 4, U12, 1, 2) ;

	// 2: U12*h1
	// h1(S1_new,S1_new') = sum{S1,S2}_[U12(S1,S2,S1_new)*h1(S1,S2,S1_new')]
	h1 = contractTensors(U12, 1, 2, h1, 1, 2) ;

	return h1 ;
}

// h12(S1_new,S2_new,S1_new',S2_new')
// = sum{S1,S2,S1',S2'}_[U12(S1,S2,S1_new)*h22(S2,S2_new,S2',S2_new')*U12(S1',S2',S1_new')*delta(S1,S1')]
// = sum{S1,S2,S2'}_[U12(S1,S2,S1_new)*h22(S2,S2_new,S2',S2_new')*U12(S1,S2',S1_new')]
Tensor DMRG::update_h12(Tensor h22, Tensor U12)
{
	// 1: h22*U12
	// h12(S2,S2_new,S2_new',S1,S1_new') = sum{S2'}_[h22(S2,S2_new,S2',S2_new')*U12(S1,S2',S1_new')]
	Tensor h12 = contractTensors(h22, 3, U12, 2) ;

	// 2: U12*h12
	// h12(S1_new,S2_new,S2_new',S1_new') = sum{S1,S2}_[U12(S1,S2,S1_new)*h12(S2,S2_new,S2_new',S1,S1_new')]
	h12 = contractTensors(U12, 1, 2, h12, 4, 1) ;

	// h12(S1_new,S2_new,S2_new',S1_new') => h12(S1_new,S2_new,S1_new',S2_new')
	h12.permute(1, 2, 4, 3) ;

	return h12 ;
}

// H12 = h1 + h2 + h12
Tensor DMRG::update_H12(Tensor h1, Tensor h2, Tensor h12)
{
	// h12(S1,S2,S1',S2')
	int dS1 = h12.dim[0] ;
	int dS2 = h12.dim[1] ;

	Tensor H12 = Tensor(dS1, dS2, dS1, dS2) ;

	for (int iS1 = 1; iS1 <= dS1; iS1 ++)
	{
		for (int iS2 = 1; iS2 <= dS2; iS2 ++)
		{
			for (int jS1 = 1; jS1 <= dS1; jS1 ++)
			{
				for (int jS2 = 1; jS2 <= dS2; jS2 ++)
				{
					H12(iS1, iS2, jS1, jS2) = h1(iS1, jS1) * delta(iS2, jS2) +
							h2(iS2, jS2) * delta(iS1, jS1) +
							h12(iS1, iS2, jS1, jS2) ;
				}
			}
		}
	}
	return H12 ;
}
//-------------------------------------------------------------------------------------------------------------------------
void DMRG::updateHamiltonian_78(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, Tensor &U78)
{
	// compute reduced density matrix of block 7,8
	Tensor densMat = computeDensMat_78(wave) ; // densMat(S7,S8,S7',S8')

	int dS7 = densMat.dim[0] ;
	int dS8 = densMat.dim[1] ;
	// densMat(S7,S8,S7',S8') => densMat((S7,S8),(S7',S8'))
	densMat.reshape(dS7 * dS8, dS7 * dS8) ;

	U78 = Tensor(dS7 * dS8, dS7 * dS8) ;
	// densMat((S7,S8),(S7',S8')) = sum{S7_new}_[U78((S7,S8),S7_new)*Lambda(S7_new)*U78((S7',S8'),S7_new)]
	Tensor Lambda = symEig(densMat, U78) ;

	double truncErr = truncateDensMat(parameter.d1, U78, Lambda) ;
	if (parameter.disp == 1)
	{
		cout << "truncation error 78: " << truncErr << endl ;
	}
	// U78((S7,S8),S7_new)
	int dS7_new = U78.dim[1] ;
	// U78((S7,S8),S7_new) => U78(S7,S8,S7_new)
	U78.reshape(dS7, dS8, dS7_new) ;

	hamiltonian.B7 = Lambda.dim[0] ;

	// h7_new(S7_new, S7_new') = sum{S7,S8,S7',S8'}_[U78(S7,S8,S7_new)*(h7(S7,S7')*delta(S8,S8')+h8(S8,S8')*delta(S7,S7')+h78(S7,S8,S7',S8'))*U78(S7',S8',S7_new')]
	hamiltonian.h7 = update_h7(hamiltonian.h7, hamiltonian.h8, hamiltonian.h78, U78) ;

	// h8_new = h8

	// h78_new(S7_new,S8_new,S7_new',S8_new')
	// = sum{S7,S8,S7',S8'}_[U78(S7,S8,S7_new)*h88(S8,S8_new,S8',S8_new')*U78(S7',S8',S7_new')*delta(S7,S7')]
	// = sum{S7,S8,S8'}_[U78(S7,S8,S7_new)*h88(S8,S8_new,S8',S8_new')*U78(S7,S8',S7_new')]
	hamiltonian.h78 = update_h12(hamiltonian.h88, U78) ;

	hamiltonian.H78 = hamiltonian.h78 ;
}
// densMat(S7,S8,S7',S8')
Tensor DMRG::computeDensMat_78(WaveFunction &wave)
{
	ReducedTensor rt = ReducedTensor(wave) ;

	// 1: T1*T2
	// densMat(Y1,X2,Y3) = sum{X1}_[T1(X1,Y1)*T2(X1,X2,Y3)]
	Tensor densMat = contractTensors(rt.T1, 1, rt.T2, 1) ;

	// 2: densMat*T3
	// densMat(Y1,Y3,Y5) = sum{X2}_[densMat(Y1,X2,Y3)*T3(X2,Y5)]
	densMat = contractTensors(densMat, 2, rt.T3, 1) ;

	// 3: densMat*T4
	// densMat(Y3,Y5,X3,Y2) = sum{Y1}_[densMat(Y1,Y3,Y5)*T4(X3,Y1,Y2)]
	densMat = contractTensors(densMat, 1, rt.T4, 2) ;

	// 4: densMat*T5
	// densMat(Y5,Y2,X4,Y4) = sum{X3,Y3}_[densMat(Y3,Y5,X3,Y2)*T5(X3,X4,Y3,Y4)]
	densMat = contractTensors(densMat, 3, 1, rt.T5, 1, 3) ;

	// 5: densMat*T6
	// densMat(Y2,Y4,Y6) = sum{X4,Y5}_[densMat(Y5,Y2,X4,Y4)*T6(X4,Y5,Y6)]
	densMat = contractTensors(densMat, 3, 1, rt.T6, 1, 2) ;

	// 6: densMat*T9
	// densMat(Y2,Y4,X6) = sum{Y6}_[densMat(Y2,Y4,Y6)*T9(X6,Y6)]
	densMat = contractTensors(densMat, 3, rt.T9, 2) ;

	// A4(x3,y1,y2,S4)
	int dx3 = wave.A4.dim[0] ;
	int dy1 = wave.A4.dim[1] ;
	int dy2 = wave.A4.dim[2] ;
	// A8(x5,x6,y4,S8)
	int dx5 = wave.A8.dim[0] ;
	int dx6 = wave.A8.dim[1] ;
	int dy4 = wave.A8.dim[2] ;

	// densMat(Y2,Y4,X6) => densMat(y2,y2',y4,y4',x6,x6')
	densMat.reshape(dy2, dy2, dy4, dy4, dx6, dx6) ;

	// 7-1: A7*A8
	// AA(y2,S7,x6,y4,S8) = sum{x5}_[A7(x5,y2,S7)*A8(x5,x6,y4,S8)]
	Tensor AA = contractTensors(wave.A7, 1, wave.A8, 1) ;

	// 7-2: AA'*densMat
	// densMat(S7',S8',y2,y4,x6) = sum{y2',x6',y4'}_[AA(y2',S7',x6',y4',S8')*densMat(y2,y2',y4,y4',x6,x6')]
	densMat = contractTensors(AA, 1, 3, 4, densMat, 2, 6, 4) ;

	// 7-3: AA*densMat
	// densMat(S7,S8,S7',S8') = sum{y2,x6,y4}_[AA(y2,S7,x6,y4,S8)*densMat(S7',S8',y2,y4,x6)]
	densMat = contractTensors(AA, 1, 3, 4, densMat, 3, 5, 4) ;

	return densMat ;
}

// h7_new(S7_new, S7_new') = sum{S7,S8,S7',S8'}_[U78(S7,S8,S7_new)*(h7(S7,S7')*delta(S8,S8')+h8(S8,S8')*delta(S7,S7')+h78(S7,S8,S7',S8'))*U78(S7',S8',S7_new')]
Tensor DMRG::update_h7(Tensor h7, Tensor h8, Tensor h78, Tensor U78)
{
	// h7_new(S7_new, S7_new') = sum{S7,S8,S7'}_[U78(S7,S8,S7_new)*h7(S7,S7')*U78(S7',S8,S7_new')]
	// 1-1: h7*U78'
	// h7(S7,S8,S7_new') = sum{S7'}_[h7(S7,S7')*U78(S7',S8,S7_new')]
	h7 = contractTensors(h7, 2, U78, 1) ;

	// 1-2: U78*h7
	// h7(S7_new, S7_new') = sum{S7,S8}_[U78(S7,S8,S7_new)*h7(S7,S8,S7_new')]
	h7 = contractTensors(U78, 1, 2, h7, 1, 2) ;

	//*****************
	// h8_new(S7_new, S7_new') = sum{S7,S8,S8'}_[U78(S7,S8,S7_new)*h8(S8,S8')*U78(S7,S8',S7_new')]
	// 2-1: h8*U78'
	// h8(S8,S7,S7_new') = sum{S8'}_[h8(S8,S8')*U78(S7,S8',S7_new')]
	h8 = contractTensors(h8, 2, U78, 2) ;

	// 2-2: U78*h8
	// h8(S7_new, S7_new') = sum{S7,S8}_[U78(S7,S8,S7_new)*h8(S8,S7,S7_new')]
	h8 = contractTensors(U78, 1, 2, h8, 2, 1) ;

	//*****************
	// h78_new(S7_new, S7_new') = sum{S7,S8,S7',S8'}_[U78(S7,S8,S7_new)*h78(S7,S8,S7',S8')*U78(S7',S8',S7_new')]
	// 3-1: h78*U78'
	// h78(S7,S8,S7_new') = sum{S7',S8'}_[h78(S7,S8,S7',S8')*U78(S7',S8',S7_new')]
	h78 = contractTensors(h78, 3, 4, U78, 1, 2) ;

	// 3-2: U78*h78
	// h78(S7_new, S7_new') = sum{S7,S8}_[U78(S7,S8,S7_new)*h78(S7,S8,S7_new')]
	h78 = contractTensors(U78, 1, 2, h78, 1, 2) ;

	//************************
	h7 = h7 + h8 + h78 ;
	return h7 ;
}
//---------------------------------------------------------------------------------------------------------------------------------------
void DMRG::updateHamiltonian_45(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, Tensor &U12, Tensor &U78)
{
	// compute reduced density matrix of block 4, 5
	Tensor densMat = computeDensMat_45(wave) ; // densMat(S4,S5,S4',S5')

	int dS4 = densMat.dim[0] ;
	int dS5 = densMat.dim[1] ;
	// densMat(S4,S5,S4',S5') => densMat((S4,S5),(S4',S5'))
	densMat.reshape(dS4 * dS5, dS4 * dS5) ;

	Tensor U45 = Tensor(dS4 * dS5, dS4 * dS5) ;
	// densMat((S4,S5),(S4',S5')) = sum{S4_new}_[U45((S4,S5),S4_new)*Lambda(S4_new)*U45((S4',S5'),S4_new)]
	Tensor Lambda = symEig(densMat, U45) ;

	double truncErr = truncateDensMat(parameter.d2, U45, Lambda) ;

	if (parameter.disp == 1)
	{
		cout << "truncation error 45: " << truncErr << endl ;
	}

	// U45((S4,S5),S4_new)
	int dS4_new = U45.dim[1] ;
	// U45((S4,S5),S4_new) => U45(S4,S5,S4_new)
	U45.reshape(dS4, dS5, dS4_new) ;

	hamiltonian.B4 = Lambda.dim[0] ;

	// h4_new(S4_new,S4_new') = sum{S4,S5,S4',S5'}_[U45(S4,S5,S4_new)*(h4(S4,S4')*delta(S5,S5')+h45(S4,S5,S4',S5'))*U45(S4',S5',S4_new')]
	hamiltonian.h4 = update_h4(hamiltonian.h4, hamiltonian.h45, U45) ;

	// h45_new(S4_new,S5_new,S4_new',S5_new')
	// = sum{S4,S5,S4'S5'}_[U45(S4,S5,S4_new)*h55(S5,S5_new,S5',S5_new')*U45(S4',S5',S4_new')*delta(S4,S4')]
	// = sum{S4,S5,S5'}_[U45(S4,S5,S4_new)*h55(S5,S5_new,S5',S5_new')*U45(S4,S5',S4_new')]
	hamiltonian.h45 =  update_h12(hamiltonian.h55, U45) ;

	// H45_new = h45_new
	hamiltonian.H45 = hamiltonian.h45 ;

	// h14_new = h14 + h25
	// = sum{S1,S2,S4,S5,S1',S2',S4',S5'}_[U12(S1,S2,S1_new)*U45(S4,S5,S4_new)*(h14(S1,S4,S1',S4') + h25(S2,S5,S2',S5'))*U12(S1',S2',S1_new')*U45(S4',S5',S4_new')]
	// = sum{S1,S4,S1',S4',S2,S5}_[U12(S1,S2,S1_new)*U45(S4,S5,S4_new)*h14(S1,S4,S1',S4')*U12(S1',S2,S1_new')*U45(S4',S5,S4_new')]
	// + sum{S2,S5,S2',S5',S1,S4}_[U12(S1,S2,S1_new)*U45(S4,S5,S4_new)*h25(S2,S5,S2',S5')*U12(S1,S2',S1_new')*U45(S4,S5',S4_new')]
	hamiltonian.h14 = update_h14(hamiltonian.h14, hamiltonian.h25, U12, U45) ;

	// H14_new = h14_new
	hamiltonian.H14 = hamiltonian.h14 ;

	// h47_new = h47 + h58
	// = sum{S4,S5,S7,S8,S4',S5',S7',S8'}_[U45(S4,S5,S4_new)*U78(S7,S8,S7_new)*(h47(S4,S7,S4',S7')+h58(S5,S8,S5',S8'))*U45(S4',S5',S4_new')*U78(S7',S8',S7_new')]
	// = sum{S4,S7,S4',S7',S5,S8}_[U45(S4,S5,S4_new)*U78(S7,S8,S7_new)*h47(S4,S7,S4',S7')*U45(S4',S5,S4_new')*U78(S7',S8,S7_new')]
	// + sum{S5,S8,S5',S8',S4,S7}_[U45(S4,S5,S4_new)*U78(S7,S8,S7_new)*h58(S5,S8,S5',S8')*U45(S4,S5',S4_new')*U78(S7,S8',S7_new')]
	hamiltonian.h47 = update_h14(hamiltonian.h47, hamiltonian.h58, U45, U78) ;

	// H47_new = h4_new + h7_new + h47_new
	hamiltonian.H47 = update_H12(hamiltonian.h4, hamiltonian.h7, hamiltonian.h47) ;

	// h44_new(S4_new,Q4_new,S4_new',Q4_new') = h44 + h55
	// = sum{S4,S5,S4',S5',Q4,Q5,Q4',Q5'}_[U45(S4,S5,S4_new)*U45(Q4,Q5,Q4_new)*(h44(S4,Q4,S4',Q4') + h55(S5,Q5,S5',Q5'))*U45(S4',S5',S4_new')*U45(Q4',Q5',Q4_new')]
	// = sum{S4,S5,S4',Q4,Q5,Q4'}_[U45(S4,S5,S4_new)*U45(Q4,Q5,Q4_new)*h44(S4,Q4,S4',Q4')*U45(S4',S5,S4_new')*U45(Q4',Q5,Q4_new')]
	// + sum{S4,S5,S5',Q4,Q5,Q5'}_[U45(S4,S5,S4_new)*U45(Q4,Q5,Q4_new)*h55(S5,Q5,S5',Q5')*U45(S4,S5',S4_new')*U45(Q4,Q5',Q4_new')]
	hamiltonian.h44 = update_h44(hamiltonian.h44, hamiltonian.h55, U45) ;

}
// densMat(S4,S5,S4',S5')
Tensor DMRG::computeDensMat_45(WaveFunction &wave)
{
	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	// A3(x2,y5,S3)
	int dx2 = wave.A3.dim[0] ;
	int dy5 = wave.A3.dim[1] ;
	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;
	// A7(x5,y2,S7)
	int dx5 = wave.A7.dim[0] ;
	int dy2 = wave.A7.dim[1] ;

	//-------------------------------------------------------------
	ReducedTensor rt = ReducedTensor(wave) ;

	// 1: T8*T9
	// densMat(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor densMat = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*densMat
	// densMat(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*densMat(X5,Y4,Y6)]
	densMat = contractTensors(rt.T7, 1, densMat, 1) ;

	// 3: T6*densMat
	// densMat(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*densMat(Y2,Y4,Y6)]
	densMat = contractTensors(rt.T6, 3, densMat, 3) ;

	// densMat(X4,Y5,Y2,Y4) => densMat(x4,x4',Y5,Y2,y4,y4')
	densMat.reshape(dx4, dx4, dy5*dy5, dy2*dy2, dy4, dy4) ;

	// 4-1: densMat*A5
	// densMat(x4',Y5,Y2,y4',x3,y3,S5) = sum{x4,y4}_[densMat(x4,x4',Y5,Y2,y4,y4')*A5(x3,x4,y3,y4,S5)]
	densMat = contractTensors(densMat, 1, 5, wave.A5, 2, 4) ;

	// 4-2: densMat*A5'
	// densMat(Y5,Y2,x3,y3,S5,x3',y3',S5') = sum{x4',y4'}_[densMat(x4',Y5,Y2,y4',x3,y3,S5)*A5(x3',x4',y3',y4',S5')]
	densMat = contractTensors(densMat, 1, 4, wave.A5, 2, 4) ;

	// densMat(Y5,Y2,x3,y3,S5,x3',y3',S5') => densMat(Y5,Y2,x3,x3',y3,y3',S5,S5')
	densMat.permute(1, 2, 3, 6, 4, 7, 5, 8) ;

	// densMat(Y5,Y2,x3,x3',y3,y3',S5,S5') => densMat(Y5,Y2,(x3,x3'),(y3,y3'),S5,S5') ~ densMat(Y5,Y2,X3,Y3,S5,S5')
	densMat.reshape(dy5*dy5, dy2*dy2, dx3*dx3, dy3*dy3, dS5, dS5) ;

	// 5: T3*densMat
	// densMat(X2,Y2,X3,Y3,S5,S5') = sum{Y5}_[T3(X2,Y5)*densMat(Y5,Y2,X3,Y3,S5,S5')]
	densMat = contractTensors(rt.T3, 2, densMat, 1) ;

	// 6: T2*densMat
	// densMat(X1,Y2,X3,S5,S5') = sum{X2,Y3}_[T2(X1,X2,Y3)*densMat(X2,Y2,X3,Y3,S5,S5')]
	densMat = contractTensors(rt.T2, 2, 3, densMat, 1, 4) ;

	// 7: T1*densMat
	// densMat(Y1,Y2,X3,S5,S5') = sum{X1}_[T1(X1,Y1)*densMat(X1,Y2,X3,S5,S5')]
	densMat = contractTensors(rt.T1, 1, densMat, 1) ;

	// densMat(Y1,Y2,X3,S5,S5') => densMat(y1,y1',y2,y2',x3,x3',S5,S5')
	densMat.reshape(dy1, dy1, dy2, dy2, dx3, dx3, dS5, dS5) ;

	// 8-1: A4'*densMat
	// densMat(S4',y1,y2,x3,S5,S5') = sum{x3',y1',y2'}_[A4(x3',y1',y2',S4')*densMat(y1,y1',y2,y2',x3,x3',S5,S5')]
	densMat = contractTensors(wave.A4, 1, 2, 3, densMat, 6, 2, 4) ;

	// 8-2: A4*densMat
	// densMat(S4,S4',S5,S5') = sum{x3,y1,y2}_[A4(x3,y1,y2,S4)*densMat(S4',y1,y2,x3,S5,S5')]
	densMat = contractTensors(wave.A4, 1, 2, 3, densMat, 4, 2, 3) ;

	// densMat(S4,S4',S5,S5') => densMat(S4,S5,S4',S5')
	densMat.permute(1, 3, 2, 4) ;

	return densMat ;
}

// h4_new(S4_new,S4_new') = sum{S4,S5,S4',S5'}_[U45(S4,S5,S4_new)*(h4(S4,S4')*delta(S5,S5')+h45(S4,S5,S4',S5'))*U45(S4',S5',S4_new')]
Tensor DMRG::update_h4(Tensor h4, Tensor h45, Tensor U45)
{
	// h4_new(S4_new,S4_new') = sum{S4,S5,S4'}_[U45(S4,S5,S4_new)*h4(S4,S4')*U45(S4',S5,S4_new')]
	// 1-1: h4*U45'
	// h4(S4,S5,S4_new') = sum{S4'}_[h4(S4,S4')*U45(S4',S5,S4_new')]
	h4 = contractTensors(h4, 2, U45, 1) ;

	// 1-2: U45*h4
	// h4(S4_new,S4_new') = sum{S4,S5}_[U45(S4,S5,S4_new)*h4(S4,S5,S4_new')]
	h4 = contractTensors(U45, 1, 2, h4, 1, 2) ;

	//**********************************************
	// h4_new(S4_new,S4_new') = sum{S4,S5,S4',S5'}_[U45(S4,S5,S4_new)*h45(S4,S5,S4',S5')*U45(S4',S5',S4_new')]
	// 2-1: h45*U45'
	// h45(S4,S5,S4_new') = sum{S4',S5'}_[h45(S4,S5,S4',S5')*U45(S4',S5',S4_new')]
	h45 = contractTensors(h45, 3, 4, U45, 1, 2) ;

	// 2-2: U45*h45
	// h45(S4_new,S4_new') = sum{S4,S5}_[U45(S4,S5,S4_new)*h45(S4,S5,S4_new')]
	h45 = contractTensors(U45, 1, 2, h45, 1, 2) ;

	//*********************************
	h4 = h4 + h45 ;
	return h4 ;
}

// h14_new = h14 + h25
// = sum{S1,S2,S4,S5,S1',S2',S4',S5'}_[U12(S1,S2,S1_new)*U45(S4,S5,S4_new)*(h14(S1,S4,S1',S4') + h25(S2,S5,S2',S5'))*U12(S1',S2',S1_new')*U45(S4',S5',S4_new')]
// = sum{S1,S4,S1',S4',S2,S5}_[U12(S1,S2,S1_new)*U45(S4,S5,S4_new)*h14(S1,S4,S1',S4')*U12(S1',S2,S1_new')*U45(S4',S5,S4_new')]
// + sum{S2,S5,S2',S5',S1,S4}_[U12(S1,S2,S1_new)*U45(S4,S5,S4_new)*h25(S2,S5,S2',S5')*U12(S1,S2',S1_new')*U45(S4,S5',S4_new')]
Tensor DMRG::update_h14(Tensor h14, Tensor h25, Tensor U12, Tensor U45)
{
	// h14_new(S1_new,S4_new,S1_new',S4_new')
	// 1-1: U12*h14
	// h14(S2,S1_new,S4,S1',S4') = sum{S1}_[U12(S1,S2,S1_new)*h14(S1,S4,S1',S4')]
	h14 = contractTensors(U12, 1, h14, 1) ;

	// 1-2: h14*U45
	// h14(S2,S1_new,S1',S4',S5,S4_new) = sum{S4}_[h14(S2,S1_new,S4,S1',S4')*U45(S4,S5,S4_new)]
	h14 = contractTensors(h14, 3, U45, 1) ;

	// 1-3: h14*U12'
	// h14(S1_new,S4',S5,S4_new,S1_new') = sum{S2,S1'}_[h14(S2,S1_new,S1',S4',S5,S4_new)*U12(S1',S2,S1_new')]
	h14 = contractTensors(h14, 1, 3, U12, 2, 1) ;

	// 1-4: h14*U45'
	// h14(S1_new,S4_new,S1_new',S4_new') = sum{S4',S5}_[h14(S1_new,S4',S5,S4_new,S1_new')*U45(S4',S5,S4_new')]
	h14 = contractTensors(h14, 2, 3, U45, 1, 2) ;

	// h25_new(S1_new,S4_new,S1_new',S4_new')
	// 2-1: U12*h25
	// h25(S1,S1_new,S5,S2',S5') = sum{S2}_[U12(S1,S2,S1_new)*h25(S2,S5,S2',S5')]
	h25 = contractTensors(U12, 2, h25, 1) ;

	// 2-2: h25*U45
	// h25(S1,S1_new,S2',S5',S4,S4_new) = sum{S5}_[h25(S1,S1_new,S5,S2',S5')*U45(S4,S5,S4_new)]
	h25 = contractTensors(h25, 3, U45, 2) ;

	// 2-3: h25*U12'
	// h25(S1_new,S5',S4,S4_new,S1_new') = sum{S1,S2'}_[h25(S1,S1_new,S2',S5',S4,S4_new)*U12(S1,S2',S1_new')]
	h25 = contractTensors(h25, 1, 3, U12, 1, 2) ;

	// 2-4: h25*U45'
	// h25(S1_new,S4_new,S1_new',S4_new') = sum{S4,S5'}_[h25(S1_new,S5',S4,S4_new,S1_new')*U45(S4,S5',S4_new')]
	h25 = contractTensors(h25, 3, 2, U45, 1, 2) ;

	h14 = h14 + h25 ;
	return h14 ;
}

// h44_new(S4_new,Q4_new,S4_new',Q4_new') = h44 + h55
// = sum{S4,S5,S4',S5',Q4,Q5,Q4',Q5'}_[U45(S4,S5,S4_new)*U45(Q4,Q5,Q4_new)*(h44(S4,Q4,S4',Q4') + h55(S5,Q5,S5',Q5'))*U45(S4',S5',S4_new')*U45(Q4',Q5',Q4_new')]
// = sum{S4,S5,S4',Q4,Q5,Q4'}_[U45(S4,S5,S4_new)*U45(Q4,Q5,Q4_new)*h44(S4,Q4,S4',Q4')*U45(S4',S5,S4_new')*U45(Q4',Q5,Q4_new')]
// + sum{S4,S5,S5',Q4,Q5,Q5'}_[U45(S4,S5,S4_new)*U45(Q4,Q5,Q4_new)*h55(S5,Q5,S5',Q5')*U45(S4,S5',S4_new')*U45(Q4,Q5',Q4_new')]
Tensor DMRG::update_h44(Tensor h44, Tensor h55, Tensor U45)
{
	// h44_new(S4_new,Q4_new,S4_new',Q4_new')
	// 1-1: U45*h44
	// h44(S5,S4_new,Q4,S4',Q4') = sum{S4}_[U45(S4,S5,S4_new)*h44(S4,Q4,S4',Q4')]
	h44 = contractTensors(U45, 1, h44, 1) ;

	// 1-2: h44*U45
	// h44(S5,S4_new,S4',Q4',Q5,Q4_new) = sum{Q4}_[h44(S5,S4_new,Q4,S4',Q4')*U45(Q4,Q5,Q4_new)]
	h44 = contractTensors(h44, 3, U45, 1) ;

	// 1-3: h44*U45'
	// h44(S4_new,Q4',Q5,Q4_new,S4_new') = sum{S5,S4'}_[h44(S5,S4_new,S4',Q4',Q5,Q4_new)*U45(S4',S5,S4_new')]
	h44 = contractTensors(h44, 1, 3, U45, 2, 1) ;

	// 1-4: h44*U45'
	// h44(S4_new,Q4_new,S4_new',Q4_new') = sum{Q4',Q5}_[h44(S4_new,Q4',Q5,Q4_new,S4_new')*U45(Q4',Q5,Q4_new')]
	h44 = contractTensors(h44, 2, 3, U45, 1, 2) ;

	// h55_new(S4_new,Q4_new,S4_new',Q4_new')
	// 2-1: U45*h55
	// h55(S4,S4_new,Q5,S5',Q5') = sum{S5}_[U45(S4,S5,S4_new)*h55(S5,Q5,S5',Q5')]
	h55 = contractTensors(U45, 2, h55, 1) ;

	// 2-2: h55*U45
	// h55(S4,S4_new,S5',Q5',Q4,Q4_new) = sum{Q5}_[h55(S4,S4_new,Q5,S5',Q5')*U45(Q4,Q5,Q4_new)]
	h55 = contractTensors(h55, 3, U45, 2) ;

	// 2-3: h55*U45'
	// h55(S4_new,Q5',Q4,Q4_new,S4_new') = sum{S4,S5'}_[h55(S4,S4_new,S5',Q5',Q4,Q4_new)*U45(S4,S5',S4_new')]
	h55 = contractTensors(h55, 1, 3, U45, 1, 2) ;

	// 2-4: h55*U45'
	// h55(S4_new,Q4_new,S4_new',Q4_new') = sum{Q4,Q5'}_[h55(S4_new,Q5',Q4,Q4_new,S4_new')*U45(Q4,Q5',Q4_new')]
	h55 = contractTensors(h55, 3, 2, U45, 1, 2) ;

	h44 = h44 + h55 ;

	return h44 ;
}

//----------------------------------------------------------------------------------------

// U(S,S_new) Lambda(S_new)
double DMRG::truncateDensMat(int dSave, Tensor &U, Tensor &Lambda)
{
	// U(S,S_new)
	int d1 = U.dim[0] ;
	int d2 = U.dim[1] ;

	if (dSave >= d2) // truncation is unnecessary
	{
		return 0.0 ;
	}
	int start = d2 - dSave + 1 ;

	U = U.subTensor(1, d1, start, d2) ;

	double L_sum = Lambda.sum() ;

	Lambda = Lambda.subTensor(start, d2) ;
	double Lcut_sum = Lambda.sum() ;

	double truncErr = 1 - Lcut_sum / L_sum ;
	return truncErr ;
}

double DMRG::delta(int a, int b)
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
// =========================================destructor==============================================
DMRG::~DMRG() {}
