/*
 * variation.cpp
 *
 *  Created on: 2011-6-27
 *      Author: zhaohuihai
 */
#include "main.h"

using namespace std ;

Variation::Variation(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave)
{
	int latticeSize = hamiltonian.length * hamiltonian.width ;
	double energy0 = 1.0 ;
	energy = 2.0 ;
	double convergeErr = abs(energy0 - energy) / abs(energy) ;
	int iterTimes = 0 ;
	while (convergeErr > parameter.tol_variation && iterTimes <= parameter.maxIter_variation)
	{

		applyVariation(parameter, hamiltonian, wave, latticeSize) ;

		convergeErr = abs(energy0 - energy) / abs(energy) ;
		//****************************************************
		cout << "energy = " << energy << endl ;
		cout << "variation convergence error = " << convergeErr << endl ;
		//****************************************************
		energy0 = energy ;
		iterTimes ++ ;
	}
	//cout << "energy = " << energy << endl ;
}

void Variation::applyVariation(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, int latticeSize)
{
	int i ;
	for (i = 0; i < 4; i ++)
	{
		// optimize A1, A3, A9, A7 successively
		optimize_A1(parameter, hamiltonian, wave, latticeSize) ;
		// optimize A2, A6, A8, A4 successively
		optimize_A2(parameter, hamiltonian, wave, latticeSize) ;

		hamiltonian.rotate() ;
		//exit(0) ;
		wave.rotate() ;
		parameter.rotate() ;
		//************************************************************

		//*************************************************************
	}
	// optimize A5
	optimize_A5(parameter, hamiltonian, wave, latticeSize) ;
}
//===========================================================================================================
void Variation::optimize_A1(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, int latticeSize)
{
	ReducedTensor rt = ReducedTensor(wave) ;
	//cout << rt.T1(1,1) << endl ;
	//exit(0) ;
	// N1((x1,y1,S1),(x1',y1',S1'))
	Tensor N1 = compute_N1(wave.A1, rt) ;
	// H1((x1,y1,S1),(x1',y1',S1'))
	Tensor H1 = compute_H1(hamiltonian, wave, rt) ;

	minimizeEnergyOneSite(N1, H1, wave.A1, latticeSize) ;


}

// N1((x1,y1,S1),(x1',y1',S1'))
Tensor Variation::compute_N1(Tensor &A1, ReducedTensor &rt)
{
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// 4: T5*m
	// m(X3,Y3,Y5,Y2) = sum{X4, Y4}_[T5(X3,X4,Y3,Y4)*m(X4,Y5,Y2,Y4)]
	m = contractTensors(rt.T5, 2, 4, m, 1, 4) ;

	// 5: T4*m
	// m(Y1,Y3,Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*m(X3,Y3,Y5,Y2)]
	m = contractTensors(rt.T4, 1, 3, m, 1, 4) ;

	// 6: T3*m
	// m(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*m(Y1,Y3,Y5)]
	m = contractTensors(rt.T3, 2, m, 3) ;

	// 7: T2*m
	// m(X1,Y1) = sum{X2,Y3}_[T2(X1,X2,Y3)*m(X2,Y1,Y3)]
	m = contractTensors(rt.T2, 2, 3, m, 1, 3) ;
	//---------------------------------------------------------------------------
	// A1(x1,y1,S1)
	int dx1 = A1.dim[0] ;
	int dy1 = A1.dim[1] ;
	int dS1 = A1.dim[2] ;

	m.reshape(dx1, dx1, dy1, dy1) ; // m(X1,Y1) => m(x1,x1',y1,y1')

	// m(x1,x1',y1,y1') => m(x1,y1,x1',y1')
	m.permute(1, 3, 2, 4) ;

	m.reshape(dx1*dy1, dx1*dy1) ; // m(x1,y1,x1',y1') => m((x1,y1),(x1',y1'))

	Tensor M = Tensor(dx1 * dy1 * dS1, dx1 * dy1 * dS1) ;
	for (int i = 1; i <= dS1; i ++)
	{
		int dv = dx1*dy1 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}

	return M ; // M((x1,y1,S1),(x1',y1',S1'))
}

// H1((x1,y1,S1),(x1',y1',S1'))
Tensor Variation::compute_H1(Hamiltonian &hamiltonian, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor H1 = compute_H1_12(hamiltonian.H12, wave, rt) ;
	H1 = H1 + compute_H1_14(hamiltonian.H14, wave, rt) ;

	H1 = H1 + compute_H1_23(hamiltonian.H23, wave, rt) ;
	H1 = H1 + compute_H1_47(hamiltonian.H47, wave, rt) ;

	H1 = H1 + compute_H1_36(hamiltonian.H36, wave, rt) ;
	H1 = H1 + compute_H1_78(hamiltonian.H78, wave, rt) ;

	H1 = H1 + compute_H1_69(hamiltonian.H69, wave, rt) ;
	H1 = H1 + compute_H1_89(hamiltonian.H89, wave, rt) ;

	H1 = H1 + compute_H1_25(hamiltonian.H25, wave, rt) ;
	H1 = H1 + compute_H1_45(hamiltonian.H45, wave, rt) ;

	H1 = H1 + compute_H1_56(hamiltonian.H56, wave, rt) ;
	H1 = H1 + compute_H1_58(hamiltonian.H58, wave, rt) ;
	return H1 ;
}
// H1_12((x1,y1,S1),(x1',y1',S1'))
Tensor Variation::compute_H1_12(Tensor &H12, WaveFunction &wave, ReducedTensor &rt)
{
	// 1: T8*T9
	// M(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor M = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*M
	// M(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*M(X5,Y4,Y6)]
	M = contractTensors(rt.T7, 1, M, 1) ;

	// 3: T6*M
	// M(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*M(Y2,Y4,Y6)]
	M = contractTensors(rt.T6, 3, M, 3) ;

	// 4: T5*M
	// M(X3,Y3,Y5,Y2) = sum{X4, Y4}_[T5(X3,X4,Y3,Y4)*M(X4,Y5,Y2,Y4)]
	M = contractTensors(rt.T5, 2, 4, M, 1, 4) ;

	// 5: T4*M
	// M(Y1,Y3,Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*M(X3,Y3,Y5,Y2)]
	M = contractTensors(rt.T4, 1, 3, M, 1, 4) ;

	// 6: T3*M
	// M(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*M(Y1,Y3,Y5)]
	M = contractTensors(rt.T3, 2, M, 3) ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;
	// A1(x1,y1,S1)
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;

	M.reshape(dx2, dx2, dy1, dy1, dy3, dy3) ; // M(X2,Y1,Y3) => M(x2,x2',y1,y1',y3,y3')

	// 7-1: A2*M
	// M(x1,S2,x2',y1,y1',y3') = sum{x2,y3}_[A2(x1,x2,y3,S2)*M(x2,x2',y1,y1',y3,y3')]
	M = contractTensors(wave.A2, 2, 3, M, 1, 5) ;

	// 7-2: A2'*M
	// M(x1',S2',x1,S2,y1,y1') = sum{x2',y3'}_[A2(x1',x2',y3',S2')*M(x1,S2,x2',y1,y1',y3')]
	M = contractTensors(wave.A2, 2, 3, M, 3, 6) ;

	// 7-3: H12*M
	// M(S1,S1',x1',x1,y1,y1') = sum{S2,S2'}_[H12(S1,S2,S1',S2')*M(x1',S2',x1,S2,y1,y1')]
	M = contractTensors(H12, 2, 4, M, 4, 2) ;

	// M(S1,S1',x1',x1,y1,y1') => M(x1,y1,S1,x1',y1',S1')
	M.permute(4, 5, 1, 3, 6, 2) ;
	M.reshape(dx1*dy1*dS1, dx1*dy1*dS1) ; // M(x1,y1,S1,x1',y1',S1') => M((x1,y1,S1),(x1',y1',S1'))

	return M ;
}
Tensor Variation::compute_H1_14(Tensor &H14, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor M ;
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H14(S1,S4,S1',S4')
	// M((y1,x1,S1),(y1',x1',S1'))
	M = compute_H1_12(H14, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;

	// M((y1,x1,S1),(y1',x1',S1')) => M(y1,x1,S1,y1',x1',S1')
	M.reshape(dy1, dx1, dS1, dy1, dx1, dS1) ;
	// M(y1,x1,S1,y1',x1',S1') => M(x1,y1,S1,x1',y1',S1')
	M.permute(2, 1, 3, 5, 4, 6) ;
	// M(x1,y1,S1,x1',y1',S1') => M((x1,y1,S1),(x1',y1',S1'))
	M.reshape(dx1*dy1*dS1, dx1*dy1*dS1) ;

	return M ;
}

Tensor Variation::compute_H1_23(Tensor &H23, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor m ;
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// 4: T5*m
	// m(X3,Y3,Y5,Y2) = sum{X4, Y4}_[T5(X3,X4,Y3,Y4)*m(X4,Y5,Y2,Y4)]
	m = contractTensors(rt.T5, 2, 4, m, 1, 4) ;

	// 5: T4*m
	// m(Y1,Y3,Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*m(X3,Y3,Y5,Y2)]
	m = contractTensors(rt.T4, 1, 3, m, 1, 4) ;

	// 6-1: A3*H23
	// m1(x2,y5,S2,S2',S3') = sum{S3}_[A3(x2,y5,S3)*H23(S2,S3,S2',S3')]
	Tensor m1 = contractTensors(wave.A3, 3, H23, 2) ;
	// 6-2: A3'*m1
	// m1(x2',y5',x2,y5,S2,S2') = sum{S3'}_[A3(x2',y5',S3')*m1(x2,y5,S2,S2',S3')]
	m1 = contractTensors(wave.A3, 3, m1, 5) ;
	// 6-3: A2*m1
	// m1(x1,y3,x2',y5',y5,S2') = sum{x2,S2}_[A2(x1,x2,y3,S2)*m1(x2',y5',x2,y5,S2,S2')]
	m1 = contractTensors(wave.A2, 2, 4, m1, 3, 5) ;
	// 6-4: A2'*m1
	// m1(x1',y3',x1,y3,y5',y5) = sum{x2',S2'}_[A2(x1',x2',y3',S2')*m1(x1,y3,x2',y5',y5,S2')]
	m1 = contractTensors(wave.A2, 2, 4, m1, 3, 6) ;

	// m1(x1',y3',x1,y3,y5',y5) => m1(x1,x1',y3,y3',y5,y5')
	m1.permute(3, 1, 4, 2, 6, 5) ;

	int dx1 = wave.A2.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dy5 = wave.A3.dim[1] ;
	int dS1 = wave.A1.dim[2] ; // A1(x1,y1,S1)
	// m1(x1,x1',y3,y3',y5,y5') => m1(x1,x1',(y3,y3'),(y5,y5')) ~ m1(x1,x1',Y3,Y5)
	m1.reshape(dx1, dx1, dy3*dy3, dy5*dy5) ;

	// 7: m1*m
	// m(x1,x1',Y1) = sum{Y3,Y5}_[m1(x1,x1',Y3,Y5)*m(Y1,Y3,Y5)]
	m = contractTensors(m1, 3, 4, m, 2, 3) ;

	// m(x1,x1',Y1) => m(x1,x1',y1,y1')
	m.reshape(dx1, dx1, dy1, dy1) ;
	// m(x1,x1',y1,y1') => m(x1,y1,x1',y1')
	m.permute(1, 3, 2, 4) ;

	m.reshape(dx1*dy1, dx1*dy1) ; // m(x1,y1,x1',y1') => m((x1,y1),(x1',y1'))


	Tensor M = Tensor(dx1 * dy1 * dS1, dx1 * dy1 * dS1) ;
	for (int i = 1; i <= dS1; i ++)
	{
		int dv = dx1*dy1 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}


	return M ; // M((x1,y1,S1),(x1',y1',S1'))
}
Tensor Variation::compute_H1_47(Tensor &H47, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor M ;
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H47(S4,S7,S4',S7') ~~ H23(S2,S3,S2',S3')
	// M((y1,x1,S1),(y1',x1',S1'))
	M = compute_H1_23(H47, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;

	// M((y1,x1,S1),(y1',x1',S1')) => M(y1,x1,S1,y1',x1',S1')
	M.reshape(dy1, dx1, dS1, dy1, dx1, dS1) ;
	// M(y1,x1,S1,y1',x1',S1') => M(x1,y1,S1,x1',y1',S1')
	M.permute(2, 1, 3, 5, 4, 6) ;
	// M(x1,y1,S1,x1',y1',S1') => M((x1,y1,S1),(x1',y1',S1'))
	M.reshape(dx1*dy1*dS1, dx1*dy1*dS1) ;

	return M ;
}

Tensor Variation::compute_H1_36(Tensor &H36, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor m ;
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2) * m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T4*m
	// m(X3,Y1,Y4,Y6) = sum{Y2}_[T4(X3,Y1,Y2) * m(Y2,Y4,Y6)]
	m = contractTensors(rt.T4, 3, m, 1) ;

	// 4: T5*m
	// m(X4,Y3,Y1,Y6) = sum{X3,Y4}_[T5(X3,X4,Y3,Y4)*m(X3,Y1,Y4,Y6)]
	m = contractTensors(rt.T5, 1, 4, m, 1, 3) ;

	// 5-1: A3*H36
	// m1(x2,y5,S6,S3',S6') = sum{S3}_[A3(x2,y5,S3)*H36(S3,S6,S3',S6')]
	Tensor m1 = contractTensors(wave.A3, 3, H36, 1) ;
	// 5-2: A3'*m1
	// m1(x2',y5',x2,y5,S6,S6') = sum{S3'}_[A3(x2',y5',S3')*m1(x2,y5,S6,S3',S6')]
	m1 = contractTensors(wave.A3, 3, m1, 4) ;
	// 5-3: A6*m1
	// m1(x4,y6,x2',y5',x2,S6') = sum{y5,S6}_[A6(x4,y5,y6,S6)*m1(x2',y5',x2,y5,S6,S6')]
	m1 = contractTensors(wave.A6, 2, 4, m1, 4, 5) ;
	// 5-4: A6'*m1
	// m1(x4',y6',x4,y6,x2',x2) = sum{y5',S6'}_[A6(x4',y5',y6',S6')*m1(x4,y6,x2',y5',x2,S6')]
	m1 = contractTensors(wave.A6, 2, 4, m1, 4, 6) ;

	// m1(x4',y6',x4,y6,x2',x2) => m1(x2,x2',x4,x4',y6,y6')
	m1.permute(6, 5, 3, 1, 4, 2) ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;
	// A3(x2,y5,S3)
	int dx2 = wave.A3.dim[0] ;
	// A6(x4,y5,y6,S6)
	int dx4 = wave.A6.dim[0] ;
	int dy6 = wave.A6.dim[2] ;

	// m1(x2,x2',x4,x4',y6,y6') => m1(X2,X4,Y6)
	m1.reshape(dx2 * dx2, dx4 * dx4, dy6 * dy6) ;

	// 6: m1*m
	// m(X2,Y3,Y1) = sum{X4,Y6}_[m1(X2,X4,Y6)*m(X4,Y3,Y1,Y6)]
	m = contractTensors(m1, 2, 3, m, 1, 4) ;

	// 7: T2*m
	// m(X1,Y1) = sum{X2,Y3}_[T2(X1,X2,Y3)*m(X2,Y3,Y1)]
	m = contractTensors(rt.T2, 2, 3, m, 1, 2) ;

	// m(X1,Y1) => m(x1,x1',y1,y1')
	m.reshape(dx1, dx1, dy1, dy1) ;
	// m(x1,x1',y1,y1') => m(x1,y1,x1',y1')
	m.permute(1, 3, 2, 4) ;

	m.reshape(dx1*dy1, dx1*dy1) ; // m(x1,y1,x1',y1') => m((x1,y1),(x1',y1'))


	Tensor M = Tensor(dx1 * dy1 * dS1, dx1 * dy1 * dS1) ;
	for (int i = 1; i <= dS1; i ++)
	{
		int dv = dx1*dy1 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}
	return M ; // M((x1,y1,S1),(x1',y1',S1'))
}
Tensor Variation::compute_H1_78(Tensor &H78, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor M ;
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H78(S7,S8,S7',S8') ~~ H36(S3,S6,S3',S6')
	// M((y1,x1,S1),(y1',x1',S1'))
	M = compute_H1_36(H78, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;

	// M((y1,x1,S1),(y1',x1',S1')) => M(y1,x1,S1,y1',x1',S1')
	M.reshape(dy1, dx1, dS1, dy1, dx1, dS1) ;
	// M(y1,x1,S1,y1',x1',S1') => M(x1,y1,S1,x1',y1',S1')
	M.permute(2, 1, 3, 5, 4, 6) ;
	// M(x1,y1,S1,x1',y1',S1') => M((x1,y1,S1),(x1',y1',S1'))
	M.reshape(dx1*dy1*dS1, dx1*dy1*dS1) ;

	return M ;
}

Tensor Variation::compute_H1_69(Tensor &H69, WaveFunction &wave, ReducedTensor &rt)
{
	// 1: T7*T8
	// m(Y2,X6,Y4) = sum{X5}_[T7(X5,Y2)*T8(X5,X6,Y4)]
	Tensor m = contractTensors(rt.T7, 1, rt.T8, 1) ;

	// 2: T4*m
	// m(X3,Y1,X6,Y4) = sum{Y2}_[T4(X3,Y1,Y2)*m(Y2,X6,Y4)]
	m = contractTensors(rt.T4, 3, m, 1) ;

	// 3: T5*m
	// m(X4,Y3,Y1,X6) = sum{X3,Y4}_[T5(X3,X4,Y3,Y4)*m(X3,Y1,X6,Y4)]
	m = contractTensors(rt.T5, 1, 4, m, 1, 4) ;

	// 4-1: A9*H69
	// m1(x6,y6,S6,S6',S9') = sum{S9}_[A9(x6,y6,S9)*H69(S6,S9,S6',S9')]
	Tensor m1 = contractTensors(wave.A9, 3, H69, 2) ;

	// 4-2: A9'*m1
	// m1(x6',y6',x6,y6,S6,S6') = sum{S9'}_[A9(x6',y6',S9')*m1(x6,y6,S6,S6',S9')]
	m1 = contractTensors(wave.A9, 3, m1, 5) ;

	// 4-3: A6*m1
	// m1(x4,y5,x6',y6',x6,S6') = sum{y6,S6}_[A6(x4,y5,y6,S6)*m1(x6',y6',x6,y6,S6,S6')]
	m1 = contractTensors(wave.A6, 3, 4, m1, 4, 5) ;

	// 4-4: A6'*m1
	// m1(x4',y5',x4,y5,x6',x6) = sum{y6',S6'}_[A6(x4',y5',y6',S6')*m1(x4,y5,x6',y6',x6,S6')]
	m1 = contractTensors(wave.A6, 3, 4, m1, 4, 6) ;

	// m1(x4',y5',x4,y5,x6',x6) => m1(x4,x4',x6,x6',y5,y5')
	m1.permute(3, 1, 6, 5, 4, 2) ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;
	// A6(x4,y5,y6,S6)
	int dx4 = wave.A6.dim[0] ;
	int dy5 = wave.A6.dim[1] ;
	// A9(x6,y6,S9)
	int dx6 = wave.A9.dim[0] ;

	// m1(x4,x4',x6,x6',y5,y5') => m1(X4,X6,Y5)
	m1.reshape(dx4*dx4, dx6*dx6, dy5*dy5) ;
	// 5: m1*m
	// m(Y5,Y3,Y1) = sum{X4,X6}_[m1(X4,X6,Y5)*m(X4,Y3,Y1,X6)]
	m = contractTensors(m1, 1, 2, m, 1, 4) ;

	// 6: T3*m
	// m(X2,Y3,Y1) = sum{Y5}_[T3(X2,Y5)*m(Y5,Y3,Y1)]
	m = contractTensors(rt.T3, 2, m, 1) ;

	// 7: T2*m
	// m(X1,Y1) = sum{X2,Y3}_[T2(X1,X2,Y3)*m(X2,Y3,Y1)]
	m = contractTensors(rt.T2, 2, 3, m, 1, 2) ;

	// m(X1,Y1) => m(x1,x1',y1,y1')
	m.reshape(dx1, dx1, dy1, dy1) ;
	// m(x1,x1',y1,y1') => m(x1,y1,x1',y1')
	m.permute(1, 3, 2, 4) ;

	m.reshape(dx1*dy1, dx1*dy1) ; // m(x1,y1,x1',y1') => m((x1,y1),(x1',y1'))


	Tensor M = Tensor(dx1 * dy1 * dS1, dx1 * dy1 * dS1) ;
	for (int i = 1; i <= dS1; i ++)
	{
		int dv = dx1*dy1 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}
	return M ;
}
Tensor Variation::compute_H1_89(Tensor &H89, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor M ;
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H89(S8,S9,S8',S9') ~~ H69(S6,S9,S6',S9')
	// M((y1,x1,S1),(y1',x1',S1'))
	M = compute_H1_69(H89, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;

	// M((y1,x1,S1),(y1',x1',S1')) => M(y1,x1,S1,y1',x1',S1')
	M.reshape(dy1, dx1, dS1, dy1, dx1, dS1) ;
	// M(y1,x1,S1,y1',x1',S1') => M(x1,y1,S1,x1',y1',S1')
	M.permute(2, 1, 3, 5, 4, 6) ;
	// M(x1,y1,S1,x1',y1',S1') => M((x1,y1,S1),(x1',y1',S1'))
	M.reshape(dx1*dy1*dS1, dx1*dy1*dS1) ;

	return M ;
}

Tensor Variation::compute_H1_25(Tensor &H25, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor m ;
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;
	// A2(x1,x2,y3,S2)
	int dx2 = wave.A2.dim[1] ;
	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;
	// A6(x4,y5,y6,S6)
	int dy5 = wave.A6.dim[1] ;
	// A7(x5,y2,S7)
	int dy2 = wave.A7.dim[1] ;

	// m(X4,Y5,Y2,Y4) => m(x4,x4',Y5,Y2,y4,y4')
	m.reshape(dx4, dx4, dy5*dy5, dy2*dy2, dy4, dy4) ;

	// 4-1: A5*m
	// m(x3,y3,S5,x4',Y5,Y2,y4') = sum{x4,y4}_[A5(x3,x4,y3,y4,S5)*m(x4,x4',Y5,Y2,y4,y4')]
	m = contractTensors(wave.A5, 2, 4, m, 1, 5) ;

	// 4-2: A5'*m
	// m(x3',y3',S5',x3,y3,S5,Y5,Y2) = sum{x4',y4'}_[A5(x3',x4',y3',y4',S5')*m(x3,y3,S5,x4',Y5,Y2,y4')]
	m = contractTensors(wave.A5, 2, 4, m, 4, 7) ;

	// m(x3',y3',S5',x3,y3,S5,Y5,Y2) => m(x3,x3',y3,y3',S5,S5',Y2,Y5)
	m.permute(4, 1, 5, 2, 6, 3, 8, 7) ;

	// m(x3,x3',y3,y3',S5,S5',Y2,Y5) => m((x3,x3'),(y3,y3'),S5,S5',Y2,Y5) ~ m(X3,Y3,S5,S5',Y2,Y5)
	m.reshape(dx3*dx3, dy3*dy3, dS5, dS5, dy2*dy2, dy5*dy5) ;

	// 5: T4*m
	// m(Y1,Y3,S5,S5',Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*m(X3,Y3,S5,S5',Y2,Y5)]
	m = contractTensors(rt.T4, 1, 3, m, 1, 5) ;

	// 6: T3*m
	// m(X2,Y1,Y3,S5,S5') = sum{Y5}_[T3(X2,Y5)*m(Y1,Y3,S5,S5',Y5)]
	m = contractTensors(rt.T3, 2, m, 5) ;

	// 7-1: A2*H25
	// m1(x1,x2,y3,S5,S2',S5') = sum{S2}_[A2(x1,x2,y3,S2)*H25(S2,S5,S2',S5')]
	Tensor m1 = contractTensors(wave.A2, 4, H25, 1) ;

	// 7-2: A2'*m1
	// m1(x1',x2',y3',x1,x2,y3,S5,S5') = sum{S2'}_[A2(x1',x2',y3',S2')*m1(x1,x2,y3,S5,S2',S5')]
	m1 = contractTensors(wave.A2, 4, m1, 5) ;

	// m1(x1',x2',y3',x1,x2,y3,S5,S5') => m1(x1,x1',x2,x2',y3,y3',S5,S5')
	m1.permute(4, 1, 5, 2, 6, 3, 7, 8) ;

	// m1(x1,x1',x2,x2',y3,y3',S5,S5') => m1((x1,x1'),(x2,x2'),(y3,y3'),S5,S5') ~ m1(X1,X2,Y3,S5,S5')
	m1.reshape(dx1*dx1, dx2*dx2, dy3*dy3, dS5, dS5) ;

	// 7-3: m1*m
	// m(X1,Y1) = sum{X2,Y3,S5,S5'}_[m1(X1,X2,Y3,S5,S5')*m(X2,Y1,Y3,S5,S5')]
	m = contractTensors(m1, 2, 3, 4, 5, m, 1, 3, 4, 5) ;

	// m(X1,Y1) => m(x1,x1',y1,y1')
	m.reshape(dx1, dx1, dy1, dy1) ;
	// m(x1,x1',y1,y1') => m(x1,y1,x1',y1')
	m.permute(1, 3, 2, 4) ;

	m.reshape(dx1*dy1, dx1*dy1) ; // m(x1,y1,x1',y1') => m((x1,y1),(x1',y1'))


	Tensor M = Tensor(dx1 * dy1 * dS1, dx1 * dy1 * dS1) ;
	for (int i = 1; i <= dS1; i ++)
	{
		int dv = dx1*dy1 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}

	return M ;
}
Tensor Variation::compute_H1_45(Tensor &H45, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor M ;
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H45(S4,S5,S4',S5') ~~ H25(S2,S5,S2',S5')
	// M((y1,x1,S1),(y1',x1',S1'))
	M = compute_H1_25(H45, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;

	// M((y1,x1,S1),(y1',x1',S1')) => M(y1,x1,S1,y1',x1',S1')
	M.reshape(dy1, dx1, dS1, dy1, dx1, dS1) ;
	// M(y1,x1,S1,y1',x1',S1') => M(x1,y1,S1,x1',y1',S1')
	M.permute(2, 1, 3, 5, 4, 6) ;
	// M(x1,y1,S1,x1',y1',S1') => M((x1,y1,S1),(x1',y1',S1'))
	M.reshape(dx1*dy1*dS1, dx1*dy1*dS1) ;

	return M ;
}

Tensor Variation::compute_H1_56(Tensor &H56, WaveFunction &wave, ReducedTensor &rt)
{
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2) * m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T4*m
	// m(X3,Y1,Y4,Y6) = sum{Y2}_[T4(X3,Y1,Y2) * m(Y2,Y4,Y6)]
	m = contractTensors(rt.T4, 3, m, 1) ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;
	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;
	// A6(x4,y5,y6,S6)
	int dy5 = wave.A6.dim[1] ;
	int dy6 = wave.A6.dim[2] ;
	int dS6 = wave.A6.dim[3] ;

	// m(X3,Y1,Y4,Y6) => m(x3,x3',Y1,y4,y4',Y6)
	m.reshape(dx3, dx3, dy1*dy1, dy4, dy4, dy6*dy6) ;

	// 4-1: A5*m
	// m(x4,y3,S5,x3',Y1,y4',Y6) = sum{x3,y4}_[A5(x3,x4,y3,y4,S5)*m(x3,x3',Y1,y4,y4',Y6)]
	m = contractTensors(wave.A5, 1, 4, m, 1, 4) ;

	// 4-2: A5'*m
	// m(x4',y3',S5',x4,y3,S5,Y1,Y6) = sum{x3',y4'}_[A5(x3',x4',y3',y4',S5')*m(x4,y3,S5,x3',Y1,y4',Y6)]
	m = contractTensors(wave.A5, 1, 4, m, 4, 6) ;

	// m(x4',y3',S5',x4,y3,S5,Y1,Y6) => m(x4,x4',y3,y3',S5,S5',Y1,Y6)
	m.permute(4, 1, 5, 2, 6, 3, 7, 8) ;

	// m(x4,x4',y3,y3',S5,S5',Y1,Y6) => m((x4,x4'),(y3,y3'),S5,S5',Y1,Y6) ~~ m(X4,Y3,S5,S5',Y1,Y6)
	m.reshape(dx4*dx4, dy3*dy3, dS5, dS5, dy1*dy1, dy6*dy6) ;

	// 5-1: A6*H56
	// m1(x4,y5,y6,S5,S5',S6') = sum{S6}_[A6(x4,y5,y6,S6)*H56(S5,S6,S5',S6')]
	Tensor m1 = contractTensors(wave.A6, 4, H56, 2) ;

	// 5-2: A6'*m1
	// m1(x4',y5',y6',x4,y5,y6,S5,S5') = sum{S6'}_[A6(x4',y5',y6',S6')*m1(x4,y5,y6,S5,S5',S6')]
	m1 = contractTensors(wave.A6, 4, m1, 6) ;

	// m1(x4',y5',y6',x4,y5,y6,S5,S5') => m1(x4,x4',y5,y5',y6,y6',S5,S5')
	m1.permute(4, 1, 5, 2, 6, 3, 7, 8) ;

	// m1(x4,x4',y5,y5',y6,y6',S5,S5') => m1((x4,x4'),(y5,y5'),(y6,y6'),S5,S5') ~~ m1(X4,Y5,Y6,S5,S5')
	m1.reshape(dx4*dx4, dy5*dy5, dy6*dy6, dS5, dS5) ;

	// 5-3: m1*m
	// m(Y5,Y3,Y1) = sum{X4,Y6,S5,S5'}_[m1(X4,Y5,Y6,S5,S5')*m(X4,Y3,S5,S5',Y1,Y6)]
	m = contractTensors(m1, 1, 3, 4, 5, m, 1, 6, 3, 4) ;

	// 6: T3*m
	// m(X2,Y3,Y1) = sum{Y5}_[T3(X2,Y5)*m(Y5,Y3,Y1)]
	m = contractTensors(rt.T3, 2, m, 1) ;

	// 7: T2*m
	// m(X1,Y1) = sum{X2,Y3}_[T2(X1,X2,Y3)*m(X2,Y3,Y1)]
	m = contractTensors(rt.T2, 2, 3, m, 1, 2) ;
	//----------------------------------------------------------------------------
	// m(X1,Y1) => m(x1,x1',y1,y1')
	m.reshape(dx1, dx1, dy1, dy1) ;
	// m(x1,x1',y1,y1') => m(x1,y1,x1',y1')
	m.permute(1, 3, 2, 4) ;

	m.reshape(dx1*dy1, dx1*dy1) ; // m(x1,y1,x1',y1') => m((x1,y1),(x1',y1'))


	Tensor M = Tensor(dx1 * dy1 * dS1, dx1 * dy1 * dS1) ;
	for (int i = 1; i <= dS1; i ++)
	{
		int dv = dx1*dy1 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}
	return M ;
}
Tensor Variation::compute_H1_58(Tensor &H58, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor M ;
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H58(S5,S8,S5',S8') ~~ H56(S5,S6,S5',S6')
	// M((y1,x1,S1),(y1',x1',S1'))
	M = compute_H1_56(H58, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;

	// M((y1,x1,S1),(y1',x1',S1')) => M(y1,x1,S1,y1',x1',S1')
	M.reshape(dy1, dx1, dS1, dy1, dx1, dS1) ;
	// M(y1,x1,S1,y1',x1',S1') => M(x1,y1,S1,x1',y1',S1')
	M.permute(2, 1, 3, 5, 4, 6) ;
	// M(x1,y1,S1,x1',y1',S1') => M((x1,y1,S1),(x1',y1',S1'))
	M.reshape(dx1*dy1*dS1, dx1*dy1*dS1) ;

	return M ;
}

//--------------------------------------------------------------------------------------------------------
void Variation::optimize_A2(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, int latticeSize)
{
	ReducedTensor rt = ReducedTensor(wave) ;
	// N2((x1,x2,y3,S2),(x1',x2',y3',S2'))
	Tensor N2 = compute_N2(wave.A2, rt) ;
	// H2((x1,x2,y3,S2),(x1',x2',y3',S2'))
	Tensor H2 = compute_H2(hamiltonian, wave, rt) ;

	minimizeEnergyOneSite(N2, H2, wave.A2, latticeSize) ;
}

Tensor Variation::compute_N2(Tensor &A2, ReducedTensor &rt)
{
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// 4: T5*m
	// m(X3,Y3,Y5,Y2) = sum{X4, Y4}_[T5(X3,X4,Y3,Y4)*m(X4,Y5,Y2,Y4)]
	m = contractTensors(rt.T5, 2, 4, m, 1, 4) ;

	// 5: T4*m
	// m(Y1,Y3,Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*m(X3,Y3,Y5,Y2)]
	m = contractTensors(rt.T4, 1, 3, m, 1, 4) ;

	// 6: T3*m
	// m(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*m(Y1,Y3,Y5)]
	m = contractTensors(rt.T3, 2, m, 3) ;

	// 7: T1*m
	// m(X1,X2,Y3) = sum{Y1}_[T1(X1,Y1)*m(X2,Y1,Y3)]
	m = contractTensors(rt.T1, 2, m, 2) ;
	// ----------------------------------------------------------------
	// A2(x1,x2,y3,S2)
	int dx1 = A2.dim[0] ;
	int dx2 = A2.dim[1] ;
	int dy3 = A2.dim[2] ;
	int dS2 = A2.dim[3] ;

	// m(X1,X2,Y3) => m(x1,x1',x2,x2',y3,y3')
	m.reshape(dx1, dx1, dx2, dx2, dy3, dy3) ;
	// m(x1,x1',x2,x2',y3,y3') => m(x1,x2,y3,x1',x2',y3')
	m.permute(1, 3, 5, 2, 4, 6) ;
	// m(x1,x2,y3,x1',x2',y3') => m((x1,x2,y3),(x1',x2',y3'))
	m.reshape(dx1*dx2*dy3, dx1*dx2*dy3) ;

	// M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	Tensor M = Tensor(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;
	for (int i = 1; i <= dS2; i ++)
	{
		int dv = dx1 * dx2 * dy3 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}
	return M ;
}

Tensor Variation::compute_H2(Hamiltonian &hamiltonian, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor H2 = compute_H2_12(hamiltonian.H12, wave, rt) ;
	H2 = H2 + compute_H2_23(hamiltonian.H23, wave, rt) ;

	H2 = H2 + compute_H2_25(hamiltonian.H25, wave, rt) ;

	H2 = H2 + compute_H2_14(hamiltonian.H14, wave, rt) ;
	H2 = H2 + compute_H2_36(hamiltonian.H36, wave, rt) ;

	H2 = H2 + compute_H2_47(hamiltonian.H47, wave, rt) ;
	H2 = H2 + compute_H2_69(hamiltonian.H69, wave, rt) ;

	H2 = H2 + compute_H2_78(hamiltonian.H78, wave, rt) ;
	H2 = H2 + compute_H2_89(hamiltonian.H89, wave, rt) ;

	H2 = H2 + compute_H2_45(hamiltonian.H45, wave, rt) ;
	H2 = H2 + compute_H2_56(hamiltonian.H56, wave, rt) ;

	H2 = H2 + compute_H2_58(hamiltonian.H58, wave, rt) ;

	return H2 ;
}

Tensor Variation::compute_H2_12(Tensor &H12, WaveFunction &wave, ReducedTensor &rt)
{
	// 1: T8*T9
	// M(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor M = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*M
	// M(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*M(X5,Y4,Y6)]
	M = contractTensors(rt.T7, 1, M, 1) ;

	// 3: T6*M
	// M(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*M(Y2,Y4,Y6)]
	M = contractTensors(rt.T6, 3, M, 3) ;

	// 4: T5*M
	// M(X3,Y3,Y5,Y2) = sum{X4, Y4}_[T5(X3,X4,Y3,Y4)*M(X4,Y5,Y2,Y4)]
	M = contractTensors(rt.T5, 2, 4, M, 1, 4) ;

	// 5: T4*M
	// M(Y1,Y3,Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*M(X3,Y3,Y5,Y2)]
	M = contractTensors(rt.T4, 1, 3, M, 1, 4) ;

	// 6: T3*M
	// M(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*M(Y1,Y3,Y5)]
	M = contractTensors(rt.T3, 2, M, 3) ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;
	// A1(x1,y1,S1)
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;

	// M(X2,Y1,Y3) => M(x2,x2',y1,y1',y3,y3')
	M.reshape(dx2, dx2, dy1, dy1, dy3, dy3) ;

	// 7-1: A1'*H12
	// M1(x1',y1',S1,S2,S2') = sum{S1'}_[A1(x1',y1',S1')*H12(S1,S2,S1',S2')]
	Tensor M1 = contractTensors(wave.A1, 3, H12, 3) ;

	// 7-2: A1*H12
	// M1(x1,y1,x1',y1',S2,S2') = sum{S1}_[A1(x1,y1,S1)*M1(x1',y1',S1,S2,S2')]
	M1 = contractTensors(wave.A1, 3, M1, 3) ;

	// 7-3: M1*M
	// M(x1,x1',S2,S2',x2,x2',y3,y3') = sum{y1,y1'}_[M1(x1,y1,x1',y1',S2,S2')*M(x2,x2',y1,y1',y3,y3')]
	M = contractTensors(M1, 2, 4, M, 3, 4) ;

	// M(x1,x1',S2,S2',x2,x2',y3,y3') => M(x1,x2,y3,S2,x1',x2',y3',S2')
	M.permute(1, 5, 7, 3, 2, 6, 8, 4) ;

	// M(x1,x2,y3,S2,x1',x2',y3',S2') => M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	M.reshape(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;

	return M ;
}
Tensor Variation::compute_H2_23(Tensor &H23, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H23(S2,S3,S2',S3') => H23(S3,S2,S3',S2') ~~ H12(S1,S2,S1',S2')
	H23.permute(2, 1, 4, 3) ;
	// M((x2,x1,y3,S2),(x2',x1',y3',S2'))
	Tensor M = compute_H2_12(H23, wave, rt) ;

	// H23(S3,S2,S3',S2') => H23(S2,S3,S2',S3')
	H23.permute(2, 1, 4, 3) ;
	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;

	// M((x2,x1,y3,S2),(x2',x1',y3',S2')) => M(x2,x1,y3,S2,x2',x1',y3',S2')
	M.reshape(dx2, dx1, dy3, dS2, dx2, dx1, dy3, dS2) ;
	// M(x2,x1,y3,S2,x2',x1',y3',S2') => M(x1,x2,y3,S2,x1',x2',y3',S2')
	M.permute(2, 1, 3, 4, 6, 5, 7, 8) ;
	// M(x1,x2,y3,S2,x1',x2',y3',S2') => M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	M.reshape(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;

	return M ;
}

Tensor Variation::compute_H2_25(Tensor &H25, WaveFunction &wave, ReducedTensor &rt)
{
	// 1: T8*T9
	// M(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor M = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*M
	// M(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*M(X5,Y4,Y6)]
	M = contractTensors(rt.T7, 1, M, 1) ;

	// 3: T6*M
	// M(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*M(Y2,Y4,Y6)]
	M = contractTensors(rt.T6, 3, M, 3) ;


	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;
	// A3(x2,y5,S3)
	// int dx2 = wave.A3.dim[0] ;
	int dy5 = wave.A3.dim[1] ;
	// A4(x3,y1,y2,S4)
	int dy2 = wave.A4.dim[2] ;
	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	// int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M(X4,Y5,Y2,Y4) => M(x4,x4',Y5,Y2,y4,y4')
	M.reshape(dx4, dx4, dy5*dy5, dy2*dy2, dy4, dy4) ;

	// 4-1: A5'*M
	// M(x3',y3',S5',x4,Y5,Y2,y4) = sum{x4',y4'}_[A5(x3',x4',y3',y4',S5')*M(x4,x4',Y5,Y2,y4,y4')]
	M = contractTensors(wave.A5, 2, 4, M, 2, 6) ;

	// 4-2: A5*M
	// M(x3,y3,S5,x3',y3',S5',Y5,Y2) = sum{x4,y4}_[A5(x3,x4,y3,y4,S5)*M(x3',y3',S5',x4,Y5,Y2,y4)]
	M = contractTensors(wave.A5, 2, 4, M, 4, 7) ;

	// M(x3,y3,S5,x3',y3',S5',Y5,Y2) => M(x3,x3',Y2,y3,y3',Y5,S5,S5')
	M.permute(1, 4, 8, 2, 5, 7, 3, 6) ;
	// M(x3,x3',Y2,y3,y3',Y5,S5,S5') => M((x3,x3'),Y2,(y3,y3'),Y5,S5,S5') ~ M(X3,Y2,Y3,Y5,S5,S5')
	M.reshape(dx3*dx3, dy2*dy2, dy3*dy3, dy5*dy5, dS5, dS5) ;

	// 5: T4*M
	// M(Y1,Y3,Y5,S5,S5') = sum{X3,Y2}_[T4(X3,Y1,Y2)*M(X3,Y2,Y3,Y5,S5,S5')]
	M = contractTensors(rt.T4, 1, 3, M, 1, 2) ;

	// 6: T3*M
	// M(X2,Y1,Y3,S5,S5') = sum{Y5}_[T3(X2,Y5)*M(Y1,Y3,Y5,S5,S5')]
	M = contractTensors(rt.T3, 2, M, 3) ;

	// 7: T1*M
	// M(X1,X2,Y3,S5,S5') = sum{Y1}_[T1(X1,Y1)*M(X2,Y1,Y3,S5,S5')]
	M = contractTensors(rt.T1, 2, M, 2) ;

	// 8: M*H25
	// M(X1,X2,Y3,S2,S2') = sum{S5,S5'}_[M(X1,X2,Y3,S5,S5')*H25(S2,S5,S2',S5')]
	M = contractTensors(M, 4, 5, H25, 2, 4) ;

	// M(X1,X2,Y3,S2,S2') => M(x1,x1',x2,x2',y3,y3',S2,S2')
	M.reshape(dx1, dx1, dx2, dx2, dy3, dy3, dS2, dS2) ;
	// M(x1,x1',x2,x2',y3,y3',S2,S2') => M(x1,x2,y3,S2,x1',x2',y3',S2')
	M.permute(1, 3, 5, 7, 2, 4, 6, 8) ;
	// M(x1,x2,y3,S2,x1',x2',y3',S2') => M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	M.reshape(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;

	return M ;
}


Tensor Variation::compute_H2_14(Tensor &H14, WaveFunction &wave, ReducedTensor &rt)
{
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// 4: T5*m
	// m(X3,Y3,Y5,Y2) = sum{X4, Y4}_[T5(X3,X4,Y3,Y4)*m(X4,Y5,Y2,Y4)]
	m = contractTensors(rt.T5, 2, 4, m, 1, 4) ;

	// 5: T3*m
	// m(X2,X3,Y3,Y2) = sum{Y5}_[T3(X2,Y5)*m(X3,Y3,Y5,Y2)]
	m = contractTensors(rt.T3, 2, m, 3) ;

	// 6-1: A1*A4
	// AA(x1,S1,x3,y2,S4) = sum{y1}_[A1(x1,y1,S1)*A4(x3,y1,y2,S4)]
	Tensor AA = contractTensors(wave.A1, 2, wave.A4, 2) ;

	// 6-2: AA*H14
	// m1(x1,x3,y2,S1',S4') = sum{S1,S4}_[AA(x1,S1,x3,y2,S4)*H14(S1,S4,S1',S4')]
	Tensor m1 = contractTensors(AA, 2, 5, H14, 1, 2) ;

	// 6-3: m1*AA'
	// m1(x1,x3,y2,x1',x3',y2') = sum{S1',S4'}_[m1(x1,x3,y2,S1',S4')*AA(x1',S1',x3',y2',S4')]
	m1 = contractTensors(m1, 4, 5, AA, 2, 5) ;

	// m1(x1,x3,y2,x1',x3',y2') => m1(x1,x1',x3,x3',y2,y2')
	m1.permute(1, 4, 2, 5, 3, 6) ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;
	// A4(x3,y1,y2,S4)
	int dx3 = wave.A4.dim[0] ;
	int dy1 = wave.A4.dim[1] ;
	int dy2 = wave.A4.dim[2] ;

	//  m1(x1,x1',x3,x3',y2,y2') => m1((x1,x1'),(x3,x3'),(y2,y2')) ~ m1(X1,X3,Y2)
	m1.reshape(dx1*dx1, dx3*dx3, dy2*dy2) ;

	// 7: m1*m
	// m(X1,X2,Y3) = sum{X3,Y2}_[m1(X1,X3,Y2)*m(X2,X3,Y3,Y2)]
	m = contractTensors(m1, 2, 3, m, 2, 4) ;

	// m(X1,X2,Y3) => m(x1,x1',x2,x2',y3,y3')
	m.reshape(dx1, dx1, dx2, dx2, dy3, dy3) ;
	// m(x1,x1',x2,x2',y3,y3') => m(x1,x2,y3,x1',x2',y3')
	m.permute(1, 3, 5, 2, 4, 6) ;
	// m(x1,x2,y3,x1',x2',y3') => m((x1,x2,y3),(x1',x2',y3'))
	m.reshape(dx1*dx2*dy3, dx1*dx2*dy3) ;

	// M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	Tensor M = Tensor(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;
	for (int i = 1; i <= dS2; i ++)
	{
		int dv = dx1 * dx2 * dy3 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}
	return M ;
}
Tensor Variation::compute_H2_36(Tensor &H36, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H36(S3,S6,S3',S6') ~~ H14(S1,S4,S1',S4')
	// M((x2,x1,y3,S2),(x2',x1',y3',S2'))
	Tensor M = compute_H2_14(H36, wave, rt) ;

	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;

	// M((x2,x1,y3,S2),(x2',x1',y3',S2')) => M(x2,x1,y3,S2,x2',x1',y3',S2')
	M.reshape(dx2, dx1, dy3, dS2, dx2, dx1, dy3, dS2) ;
	// M(x2,x1,y3,S2,x2',x1',y3',S2') => M(x1,x2,y3,S2,x1',x2',y3',S2')
	M.permute(2, 1, 3, 4, 6, 5, 7, 8) ;
	// M(x1,x2,y3,S2,x1',x2',y3',S2') => M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	M.reshape(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;

	return M ;
}


Tensor Variation::compute_H2_47(Tensor &H47, WaveFunction &wave, ReducedTensor &rt)
{
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T6*m
	// m(X4,Y5,X5,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// 3: T5*m
	// m(X3,Y3,Y5,X5) = sum{X4,Y4}_[T5(X3,X4,Y3,Y4)*m(X4,Y5,X5,Y4)]
	m = contractTensors(rt.T5, 2, 4, m, 1, 4) ;

	// 4-1: A4*A7
	// AA(x3,y1,S4,x5,S7) = sum{y2}_[A4(x3,y1,y2,S4)*A7(x5,y2,S7)]
	Tensor AA = contractTensors(wave.A4, 3, wave.A7, 2) ;

	// 4-2: AA*H47
	// m1(x3,y1,x5,S4',S7') = sum{S4,S7}_[AA(x3,y1,S4,x5,S7)*H47(S4,S7,S4',S7')]
	Tensor m1 = contractTensors(AA, 3, 5, H47, 1, 2) ;

	// 4-3: m1*AA'
	// m1(x3,y1,x5,x3',y1',x5') = sum{S4',S7'}_[m1(x3,y1,x5,S4',S7')*AA(x3',y1',S4',x5',S7')]
	m1 = contractTensors(m1, 4, 5, AA, 3, 5) ;

	// m1(x3,y1,x5,x3',y1',x5') => m1(x3,x3',x5,x5',y1,y1')
	m1.permute(1, 4, 3, 6, 2, 5) ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;
	// A4(x3,y1,y2,S4)
	int dx3 = wave.A4.dim[0] ;
	int dy1 = wave.A4.dim[1] ;
	int dy2 = wave.A4.dim[2] ;
	// A7(x5,y2,S7)
	int dx5 = wave.A7.dim[0] ;

	// m1(x3,x3',x5,x5',y1,y1') => m1((x3,x3'),(x5,x5'),(y1,y1')) ~ m1(X3,X5,Y1)
	m1.reshape(dx3*dx3, dx5*dx5, dy1*dy1) ;

	// 5: m1*m
	// m(Y1,Y3,Y5) = sum{X3,X5}_[m1(X3,X5,Y1)*m(X3,Y3,Y5,X5)]
	m = contractTensors(m1, 1, 2, m, 1, 4) ;

	// 6: T3*m
	// m(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*m(Y1,Y3,Y5)]
	m = contractTensors(rt.T3, 2, m, 3) ;

	// 7: T1*m
	// m(X1,X2,Y3) = sum{Y1}_[T1(X1,Y1)*m(X2,Y1,Y3)]
	m = contractTensors(rt.T1, 2, m, 2) ;
	// ----------------------------------------------------------------
	// m(X1,X2,Y3) => m(x1,x1',x2,x2',y3,y3')
	m.reshape(dx1, dx1, dx2, dx2, dy3, dy3) ;
	// m(x1,x1',x2,x2',y3,y3') => m(x1,x2,y3,x1',x2',y3')
	m.permute(1, 3, 5, 2, 4, 6) ;
	// m(x1,x2,y3,x1',x2',y3') => m((x1,x2,y3),(x1',x2',y3'))
	m.reshape(dx1*dx2*dy3, dx1*dx2*dy3) ;

	// M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	Tensor M = Tensor(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;
	for (int i = 1; i <= dS2; i ++)
	{
		int dv = dx1 * dx2 * dy3 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}

	return M ;
}
Tensor Variation::compute_H2_69(Tensor &H69, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H69(S6,S9,S6',S9') ~~ H47(S4,S7,S4',S7')
	// M((x2,x1,y3,S2),(x2',x1',y3',S2'))
	Tensor M = compute_H2_47(H69, wave, rt) ;

	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;

	// M((x2,x1,y3,S2),(x2',x1',y3',S2')) => M(x2,x1,y3,S2,x2',x1',y3',S2')
	M.reshape(dx2, dx1, dy3, dS2, dx2, dx1, dy3, dS2) ;
	// M(x2,x1,y3,S2,x2',x1',y3',S2') => M(x1,x2,y3,S2,x1',x2',y3',S2')
	M.permute(2, 1, 3, 4, 6, 5, 7, 8) ;
	// M(x1,x2,y3,S2,x1',x2',y3',S2') => M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	M.reshape(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;

	return M ;
}


Tensor Variation::compute_H2_78(Tensor &H78, WaveFunction &wave, ReducedTensor &rt)
{
	// 1-1: A7*A8
	// AA(y2,S7,x6,y4,S8) = sum{x5}_[A7(x5,y2,S7)*A8(x5,x6,y4,S8)]
	Tensor AA = contractTensors(wave.A7, 1, wave.A8, 1) ;

	// 1-2: AA*H78
	// m(y2,x6,y4,S7',S8') = sum{S7,S8}_[AA(y2,S7,x6,y4,S8)*H78(S7,S8,S7',S8')]
	Tensor m = contractTensors(AA, 2, 5, H78, 1, 2) ;

	// 1-3: m*AA'
	// m(y2,x6,y4,y2',x6',y4') = sum{S7',S8'}_[m(y2,x6,y4,S7',S8')*AA(y2',S7',x6',y4',S8')]
	m = contractTensors(m, 4, 5, AA, 2, 5) ;

	// m(y2,x6,y4,y2',x6',y4') => m(x6,x6',y2,y2',y4,y4')
	m.permute(2, 5, 1, 4, 3, 6) ;

	// A7(x5,y2,S7)
	int dy2 = wave.A7.dim[1] ;
	// A8(x5,x6,y4,S8)
	int dx6 = wave.A8.dim[1] ;
	int dy4 = wave.A8.dim[2] ;

	// m(x6,x6',y2,y2',y4,y4') => m((x6,x6'),(y2,y2'),(y4,y4')) ~ m(X6,Y2,Y4)
	m.reshape(dx6*dx6, dy2*dy2, dy4*dy4) ;

	// 2: m*T9
	// m(Y2,Y4,Y6) = sum{X6}_[m(X6,Y2,Y4)*T9(X6,Y6)]
	m = contractTensors(m, 1, rt.T9, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// 4: T5*m
	// m(X3,Y3,Y5,Y2) = sum{X4, Y4}_[T5(X3,X4,Y3,Y4)*m(X4,Y5,Y2,Y4)]
	m = contractTensors(rt.T5, 2, 4, m, 1, 4) ;

	// 5: T4*m
	// m(Y1,Y3,Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*m(X3,Y3,Y5,Y2)]
	m = contractTensors(rt.T4, 1, 3, m, 1, 4) ;

	// 6: T3*m
	// m(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*m(Y1,Y3,Y5)]
	m = contractTensors(rt.T3, 2, m, 3) ;

	// 7: T1*m
	// m(X1,X2,Y3) = sum{Y1}_[T1(X1,Y1)*m(X2,Y1,Y3)]
	m = contractTensors(rt.T1, 2, m, 2) ;
	// ----------------------------------------------------------------
	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;

	// m(X1,X2,Y3) => m(x1,x1',x2,x2',y3,y3')
	m.reshape(dx1, dx1, dx2, dx2, dy3, dy3) ;
	// m(x1,x1',x2,x2',y3,y3') => m(x1,x2,y3,x1',x2',y3')
	m.permute(1, 3, 5, 2, 4, 6) ;
	// m(x1,x2,y3,x1',x2',y3') => m((x1,x2,y3),(x1',x2',y3'))
	m.reshape(dx1*dx2*dy3, dx1*dx2*dy3) ;

	// M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	Tensor M = Tensor(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;
	for (int i = 1; i <= dS2; i ++)
	{
		int dv = dx1 * dx2 * dy3 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}
	return M ;
}
Tensor Variation::compute_H2_89(Tensor &H89, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H89(S8,S9,S8',S9') => H89(S9,S8,S9',S8') ~~ H78(S7,S8,S7',S8')
	H89.permute(2, 1, 4, 3) ;
	// M((x2,x1,y3,S2),(x2',x1',y3',S2'))
	Tensor M = compute_H2_78(H89, wave, rt) ;

	// H89(S9,S8,S9',S8') => H89(S8,S9,S8',S9')
	H89.permute(2, 1, 4, 3) ;
	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;

	// M((x2,x1,y3,S2),(x2',x1',y3',S2')) => M(x2,x1,y3,S2,x2',x1',y3',S2')
	M.reshape(dx2, dx1, dy3, dS2, dx2, dx1, dy3, dS2) ;
	// M(x2,x1,y3,S2,x2',x1',y3',S2') => M(x1,x2,y3,S2,x1',x2',y3',S2')
	M.permute(2, 1, 3, 4, 6, 5, 7, 8) ;
	// M(x1,x2,y3,S2,x1',x2',y3',S2') => M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	M.reshape(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;

	return M ;
}


Tensor Variation::compute_H2_45(Tensor &H45, WaveFunction &wave, ReducedTensor &rt)
{
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;
	// A4(x3,y1,y2,S4)
	int dx3 = wave.A4.dim[0] ;
	int dy1 = wave.A4.dim[1] ;
	int dy2 = wave.A4.dim[2] ;
	// A5(x3,x4,y3,y4,S5)
	//int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	//int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;
	// A6(x4,y5,y6,S6)
	int dy5 = wave.A6.dim[1] ;

	// m(X4,Y5,Y2,Y4) => m(x4,x4',Y5,Y2,y4,y4')
	m.reshape(dx4, dx4, dy5*dy5, dy2*dy2, dy4, dy4) ;

	// 4-1: A5'*m
	// m(x3',y3',S5',x4,Y5,Y2,y4) = sum{x4',y4'}_[A5(x3',x4',y3',y4',S5')*m(x4,x4',Y5,Y2,y4,y4')]
	m = contractTensors(wave.A5, 2, 4, m, 2, 6) ;

	// 4-2: A5*m
	// m(x3,y3,S5,x3',y3',S5',Y5,Y2) = sum{x4,y4}_[A5(x3,x4,y3,y4,S5)*m(x3',y3',S5',x4,Y5,Y2,y4)]
	m = contractTensors(wave.A5, 2, 4, m, 4, 7) ;

	// m(x3,y3,S5,x3',y3',S5',Y5,Y2) => m(x3,x3',Y2,y3,y3',Y5,S5,S5')
	m.permute(1, 4, 8, 2, 5, 7, 3, 6) ;

	// m(x3,x3',Y2,y3,y3',Y5,S5,S5') => m((x3,x3'),Y2,(y3,y3'),Y5,S5,S5') ~ m(X3,Y2,Y3,Y5,S5,S5')
	m.reshape(dx3*dx3, dy2*dy2, dy3*dy3, dy5*dy5, dS5, dS5) ;

	// 5-1: A4'*H45
	// m1(x3',y1',y2',S4,S5,S5') = sum{S4'}_[A4(x3',y1',y2',S4')*H45(S4,S5,S4',S5')]
	Tensor m1 = contractTensors(wave.A4, 4, H45, 3) ;

	// 5-2: A4*m1
	// m1(x3,y1,y2,x3',y1',y2',S5,S5') = sum{S4}_[A4(x3,y1,y2,S4)*m1(x3',y1',y2',S4,S5,S5')]
	m1 = contractTensors(wave.A4, 4, m1, 4) ;

	// m1(x3,y1,y2,x3',y1',y2',S5,S5') => m1(x3,x3',y1,y1',y2,y2',S5,S5')
	m1.permute(1, 4, 2, 5, 3, 6, 7, 8) ;

	// m1(x3,x3',y1,y1',y2,y2',S5,S5') => m1((x3,x3'),(y1,y1'),(y2,y2'),S5,S5') ~ m1(X3,Y1,Y2,S5,S5')
	m1.reshape(dx3*dx3, dy1*dy1, dy2*dy2, dS5, dS5) ;

	// 5-3: m1*m
	// m(Y1,Y3,Y5) = sum{X3,Y2,S5,S5'}_[m1(X3,Y1,Y2,S5,S5')*m(X3,Y2,Y3,Y5,S5,S5')]
	m = contractTensors(m1, 1, 3, 4, 5, m, 1, 2, 5, 6) ;

	// 6: T3*m
	// m(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*m(Y1,Y3,Y5)]
	m = contractTensors(rt.T3, 2, m, 3) ;

	// 7: T1*m
	// m(X1,X2,Y3) = sum{Y1}_[T1(X1,Y1)*m(X2,Y1,Y3)]
	m = contractTensors(rt.T1, 2, m, 2) ;
	// ----------------------------------------------------------------

	// m(X1,X2,Y3) => m(x1,x1',x2,x2',y3,y3')
	m.reshape(dx1, dx1, dx2, dx2, dy3, dy3) ;
	// m(x1,x1',x2,x2',y3,y3') => m(x1,x2,y3,x1',x2',y3')
	m.permute(1, 3, 5, 2, 4, 6) ;
	// m(x1,x2,y3,x1',x2',y3') => m((x1,x2,y3),(x1',x2',y3'))
	m.reshape(dx1*dx2*dy3, dx1*dx2*dy3) ;

	// M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	Tensor M = Tensor(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;
	for (int i = 1; i <= dS2; i ++)
	{
		int dv = dx1 * dx2 * dy3 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}
	return M ;
}
Tensor Variation::compute_H2_56(Tensor &H56, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H56(S5,S6,S5',S6') => H56(S6,S5,S6',S5') ~~ H45(S4,S5,S4',S5')
	H56.permute(2, 1, 4, 3) ;
	// M((x2,x1,y3,S2),(x2',x1',y3',S2'))
	Tensor M = compute_H2_45(H56, wave, rt) ;

	// H56(S6,S5,S6',S5') => H56(S5,S6,S5',S6')
	H56.permute(2, 1, 4, 3) ;
	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;

	// M((x2,x1,y3,S2),(x2',x1',y3',S2')) => M(x2,x1,y3,S2,x2',x1',y3',S2')
	M.reshape(dx2, dx1, dy3, dS2, dx2, dx1, dy3, dS2) ;
	// M(x2,x1,y3,S2,x2',x1',y3',S2') => M(x1,x2,y3,S2,x1',x2',y3',S2')
	M.permute(2, 1, 3, 4, 6, 5, 7, 8) ;
	// M(x1,x2,y3,S2,x1',x2',y3',S2') => M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	M.reshape(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;

	return M ;
}


Tensor Variation::compute_H2_58(Tensor &H58, WaveFunction &wave, ReducedTensor &rt)
{
	// A2(x1,x2,y3,S2)
	int dx1 = wave.A2.dim[0] ;
	int dx2 = wave.A2.dim[1] ;
	int dy3 = wave.A2.dim[2] ;
	int dS2 = wave.A2.dim[3] ;
	// A4(x3,y1,y2,S4)
	int dx3 = wave.A4.dim[0] ;
	int dy1 = wave.A4.dim[1] ;
	int dy2 = wave.A4.dim[2] ;
	// A5(x3,x4,y3,y4,S5)
	//int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	//int dy3 = wave.A5.dim[2] ;
	//int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;
	// A6(x4,y5,y6,S6)
	int dy5 = wave.A6.dim[1] ;
	// A8(x5,x6,y4,S8)
	int dx5 = wave.A8.dim[0] ;
	int dx6 = wave.A8.dim[1] ;
	int dy4 = wave.A8.dim[2] ;

	// 1: T6*T9
	// m(X4,Y5,X6) = sum{Y6}_[T6(X4,Y5,Y6)*T9(X6,Y6)]
	Tensor m = contractTensors(rt.T6, 3, rt.T9, 2) ;

	// m(X4,Y5,X6) => m(X4,Y5,x6,x6')
	m.reshape(dx4*dx4, dy5*dy5, dx6, dx6) ;

	// 2-1: A8'*H58
	// m1(x5',x6',y4',S5,S8,S5') = sum{S8'}_[A8(x5',x6',y4',S8')*H58(S5,S8,S5',S8')]
	Tensor m1 = contractTensors(wave.A8, 4, H58, 4) ;

	// 2-2: A8 *m1
	// m1(x5,x6,y4,x5',x6',y4',S5,S5') = sum{S8}_[A8(x5,x6,y4,S8)*m1(x5',x6',y4',S5,S8,S5')]
	m1 = contractTensors(wave.A8, 4, m1, 5) ;

	// 2-3: m1*m
	// m(x5,y4,x5',y4',S5,S5',X4,Y5) = sum{x6,x6'}_[m1(x5,x6,y4,x5',x6',y4',S5,S5')*m(X4,Y5,x6,x6')]
	m = contractTensors(m1, 2, 5, m, 3, 4) ;

	// m(x5,y4,x5',y4',S5,S5',X4,Y5) => m(X4,x5,x5',y4,y4',Y5,S5,S5')
	m.permute(7, 1, 3, 2, 4, 8, 5, 6) ;

	// m(X4,x5,x5',y4,y4',Y5,S5,S5') => m(x4,x4',(x5,x5'),y4,y4',Y5,S5,S5') ~ m(x4,x4',X5,y4,y4',Y5,S5,S5')
	m.reshape(dx4, dx4, dx5*dx5, dy4, dy4, dy5*dy5, dS5, dS5) ;

	// 3-1: A5'*m
	// m(x3',y3',x4,X5,y4,Y5,S5) = sum{x4',y4',S5'}_[A5(x3',x4',y3',y4',S5')*m(x4,x4',X5,y4,y4',Y5,S5,S5')]
	m = contractTensors(wave.A5, 2, 4, 5, m, 2, 5, 8) ;

	// 3-2: A5*m
	// m(x3,y3,x3',y3',X5,Y5) = sum{x4,y4,S5}_[A5(x3,x4,y3,y4,S5)*m(x3',y3',x4,X5,y4,Y5,S5)]
	m = contractTensors(wave.A5, 2, 4, 5, m, 3, 5, 7) ;

	// m(x3,y3,x3',y3',X5,Y5) => m(x3,x3',X5,y3,y3',Y5)
	m.permute(1, 3, 5, 2, 4, 6) ;

	// m(x3,x3',X5,y3,y3',Y5) => m((x3,x3'),X5,(y3,y3'),Y5) ~ m(X3,X5,Y3,Y5)
	m.reshape(dx3*dx3, dx5*dx5, dy3*dy3, dy5*dy5) ;

	// 4: T7*m
	// m(Y2,X3,Y3,Y5) = sum{X5}_[T7(X5,Y2)*m(X3,X5,Y3,Y5)]
	m = contractTensors(rt.T7, 1, m, 2) ;

	// 5: T4*m
	// m(Y1,Y3,Y5) = sum{X3,Y2}_[T4(X3,Y1,Y2)*m(Y2,X3,Y3,Y5)]
	m = contractTensors(rt.T4, 1, 3, m, 2, 1) ;

	// 6: T3*m
	// m(X2,Y1,Y3) = sum{Y5}_[T3(X2,Y5)*m(Y1,Y3,Y5)]
	m = contractTensors(rt.T3, 2, m, 3) ;

	// 7: T1*m
	// m(X1,X2,Y3) = sum{Y1}_[T1(X1,Y1)*m(X2,Y1,Y3)]
	m = contractTensors(rt.T1, 2, m, 2) ;
	// ----------------------------------------------------------------

	// m(X1,X2,Y3) => m(x1,x1',x2,x2',y3,y3')
	m.reshape(dx1, dx1, dx2, dx2, dy3, dy3) ;
	// m(x1,x1',x2,x2',y3,y3') => m(x1,x2,y3,x1',x2',y3')
	m.permute(1, 3, 5, 2, 4, 6) ;
	// m(x1,x2,y3,x1',x2',y3') => m((x1,x2,y3),(x1',x2',y3'))
	m.reshape(dx1*dx2*dy3, dx1*dx2*dy3) ;

	// M((x1,x2,y3,S2),(x1',x2',y3',S2'))
	Tensor M = Tensor(dx1*dx2*dy3*dS2, dx1*dx2*dy3*dS2) ;
	for (int i = 1; i <= dS2; i ++)
	{
		int dv = dx1 * dx2 * dy3 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}
	return M ;
}

//--------------------------------------------------------------------------------------------------------
void Variation::optimize_A5(Parameter &parameter, Hamiltonian &hamiltonian, WaveFunction &wave, int latticeSize)
{
	ReducedTensor rt = ReducedTensor(wave) ;

	// N5((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	Tensor N5 = compute_N5(wave.A5, rt) ;

	// H5((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	Tensor H5 = compute_H5(hamiltonian, wave, rt) ;
	//exit(0) ;
	minimizeEnergyOneSite(N5, H5, wave.A5, latticeSize) ;
}
// N5((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
Tensor Variation::compute_N5(Tensor &A5, ReducedTensor &rt)
{
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// 4: T1*T2
	// m1(Y1,X2,Y3) = sum{X1}_[T1(X1,Y1)*T2(X1,X2,Y3)]
	Tensor m1 = contractTensors(rt.T1, 1, rt.T2, 1) ;

	// 5: m1*T3
	// m1(Y1,Y3,Y5) = sum{X2}_[m1(Y1,X2,Y3)*T3(X2,Y5)]
	m1 = contractTensors(m1, 2, rt.T3, 1) ;

	// 6: m1*T4
	// m1(Y3,Y5,X3,Y2) = sum{Y1}_[m1(Y1,Y3,Y5)*T4(X3,Y1,Y2)]
	m1 = contractTensors(m1, 1, rt.T4, 2) ;

	// 7: m1*m
	// m(Y3,X3,X4,Y4) = sum{Y5,Y2}_[m1(Y3,Y5,X3,Y2)*m(X4,Y5,Y2,Y4)]
	m = contractTensors(m1, 2, 4, m, 2, 3) ;
	//-------------------------------------------------------------------
	// A5(x3,x4,y3,y4,S5)
	int dx3 = A5.dim[0] ;
	int dx4 = A5.dim[1] ;
	int dy3 = A5.dim[2] ;
	int dy4 = A5.dim[3] ;
	int dS5 = A5.dim[4] ;

	// m(Y3,X3,X4,Y4) => m(y3,y3',x3,x3',x4,x4',y4,y4')
	m.reshape(dy3, dy3, dx3, dx3, dx4, dx4, dy4, dy4) ;

	// m(y3,y3',x3,x3',x4,x4',y4,y4') => m(x3,x4,y3,y4,x3',x4',y3',y4')
	m.permute(3, 5, 1, 7, 4, 6, 2, 8) ;

	// m(x3,x4,y3,y4,x3',x4',y3',y4') => m((x3,x4,y3,y4),(x3',x4',y3',y4'))
	m.reshape(dx3*dx4*dy3*dy4, dx3*dx4*dy3*dy4) ;

	// M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	Tensor M = Tensor(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;
	for (int i = 1; i <= dS5; i ++)
	{
		int dv = dx3*dx4*dy3*dy4 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}

	return M ;
}

Tensor Variation::compute_H5(Hamiltonian &hamiltonian, WaveFunction &wave, ReducedTensor &rt)
{
	Tensor H5 = compute_H5_12(hamiltonian.H12, wave, rt) ;
	//exit(0) ;
	H5 = H5 + compute_H5_23(hamiltonian.H23, wave, rt) ;
	H5 = H5 + compute_H5_36(hamiltonian.H36, wave, rt) ;
	H5 = H5 + compute_H5_69(hamiltonian.H69, wave, rt) ;
	H5 = H5 + compute_H5_14(hamiltonian.H14, wave, rt) ;
	H5 = H5 + compute_H5_47(hamiltonian.H47, wave, rt) ;
	H5 = H5 + compute_H5_78(hamiltonian.H78, wave, rt) ;
	H5 = H5 + compute_H5_89(hamiltonian.H89, wave, rt) ;

	H5 = H5 + compute_H5_25(hamiltonian.H25, wave, rt) ;
	H5 = H5 + compute_H5_45(hamiltonian.H45, wave, rt) ;
	H5 = H5 + compute_H5_56(hamiltonian.H56, wave, rt) ;
	H5 = H5 + compute_H5_58(hamiltonian.H58, wave, rt) ;

	return H5 ;
}

Tensor Variation::compute_H5_12(Tensor &H12, WaveFunction &wave, ReducedTensor &rt)
{
	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;
	// A3(x2,y5,S3)
	int dx2 = wave.A3.dim[0] ;
	int dy5 = wave.A3.dim[1] ;
	int dS3 = wave.A3.dim[2] ;
	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;
	//----------------------------------------------------------
	// 1: T8*T9
	// m(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor m = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*m
	// m(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*m(X5,Y4,Y6)]
	m = contractTensors(rt.T7, 1, m, 1) ;

	// 3: T6*m
	// m(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*m(Y2,Y4,Y6)]
	m = contractTensors(rt.T6, 3, m, 3) ;

	// 4-1: A1*A2
	// AA(y1,S1,x2,y3,S2) = sum{x1}_[A1(x1,y1,S1)*A2(x1,x2,y3,S2)]
	Tensor AA = contractTensors(wave.A1, 1, wave.A2, 1) ;

	// 4-2: AA*H12
	// m1(y1,x2,y3,S1',S2') = sum{S1,S2}_[AA(y1,S1,x2,y3,S2)*H12(S1,S2,S1',S2')]
	Tensor m1 = contractTensors(AA, 2, 5, H12, 1, 2) ;

	// 4-3: m1*AA'
	// m1(y1,x2,y3,y1',x2',y3') = sum{S1',S2'}_[m1(y1,x2,y3,S1',S2')*AA(y1',S1',x2',y3',S2')]
	m1 = contractTensors(m1, 4, 5, AA, 2, 5) ;

	// m1(y1,x2,y3,y1',x2',y3') => m1(y1,y1',x2,x2',y3,y3')
	m1.permute(1, 4, 2, 5, 3, 6) ;

	// m1(y1,y1',x2,x2',y3,y3') => m1((y1,y1'),(x2,x2'),(y3,y3')) => m1(Y1,X2,Y3)
	m1.reshape(dy1*dy1, dx2*dx2, dy3*dy3) ;

	// 5: m1*T3
	// m1(Y1,Y3,Y5) = sum{X2}_[m1(Y1,X2,Y3)*T3(X2,Y5)]
	m1 = contractTensors(m1, 2, rt.T3, 1) ;

	// 6: m1*T4
	// m1(Y3,Y5,X3,Y2) = sum{Y1}_[m1(Y1,Y3,Y5)*T4(X3,Y1,Y2)]
	m1 = contractTensors(m1, 1, rt.T4, 2) ;

	// 7: m1*m
	// m(Y3,X3,X4,Y4) = sum{Y5,Y2}_[m1(Y3,Y5,X3,Y2)*m(X4,Y5,Y2,Y4)]
	m = contractTensors(m1, 2, 4, m, 2, 3) ;
	//-------------------------------------------------------------------
	// m(Y3,X3,X4,Y4) => m(y3,y3',x3,x3',x4,x4',y4,y4')
	m.reshape(dy3, dy3, dx3, dx3, dx4, dx4, dy4, dy4) ;

	// m(y3,y3',x3,x3',x4,x4',y4,y4') => m(x3,x4,y3,y4,x3',x4',y3',y4')
	m.permute(3, 5, 1, 7, 4, 6, 2, 8) ;

	// m(x3,x4,y3,y4,x3',x4',y3',y4') => m((x3,x4,y3,y4),(x3',x4',y3',y4'))
	m.reshape(dx3*dx4*dy3*dy4, dx3*dx4*dy3*dy4) ;

	// M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	Tensor M = Tensor(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;
	for (int i = 1; i <= dS5; i ++)
	{
		int dv = dx3*dx4*dy3*dy4 ;
		size_t start = (i - 1) * dv + 1 ;
		size_t end = i * dv ;
		M.value[M.getSlice(start, end, start, end)] = m.value ;
	}

	return M ;
}

// 12 -> 14
Tensor Variation::compute_H5_14(Tensor &H14, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H14(S1,S4,S1',S4') ~~ H12(S1,S2,S1',S2')
	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5'))
	Tensor M = compute_H5_12(H14, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5')) => M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')
	M.reshape(dy3, dy4, dx3, dx4, dS5, dy3, dy4, dx3, dx4, dS5) ;
	// M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')   =>   M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(3, 4, 1, 2, 5, 8, 9, 6, 7, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}
// 12 -> 14 -> 36
Tensor Variation::compute_H5_36(Tensor &H36, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H36(S3,S6,S3',S6') ~~ H14(S1,S4,S1',S4')
	// M((x4,x3,y3,y4,S5),(x4',x3',y3',y4',S5'))
	Tensor M = compute_H5_14(H36, wave, rt) ;

	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((x4,x3,y3,y4,S5),(x4',x3',y3',y4',S5')) => M(x4,x3,y3,y4,S5,x4',x3',y3',y4',S5')
	M.reshape(dx4, dx3, dy3, dy4, dS5, dx4, dx3, dy3, dy4, dS5) ;
	// M(x4,x3,y3,y4,S5,x4',x3',y3',y4',S5')   =>   M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(2, 1, 3, 4, 5, 7, 6, 8, 9, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}
// 12 -> 14 -> 36 -> 78
Tensor Variation::compute_H5_78(Tensor &H78, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H78(S7,S8,S7',S8') ~~ H36(S3,S6,S3',S6')
	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5'))
	Tensor M = compute_H5_36(H78, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5')) => M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')
	M.reshape(dy3, dy4, dx3, dx4, dS5, dy3, dy4, dx3, dx4, dS5) ;
	// M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')   =>   M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(3, 4, 1, 2, 5, 8, 9, 6, 7, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}

// 12 -> 23
Tensor Variation::compute_H5_23(Tensor &H23, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H23(S2,S3,S2',S3') => H23(S3,S2,S3',S2') ~~ H12(S1,S2,S1',S2')
	H23.permute(2, 1, 4, 3) ;
	// M((x4,x3,y3,y4,S5),(x4',x3',y3',y4',S5'))
	Tensor M = compute_H5_12(H23, wave, rt) ;

	// H23(S3,S2,S3',S2') => H23(S2,S3,S2',S3')
	H23.permute(2, 1, 4, 3) ;
	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((x4,x3,y3,y4,S5),(x4',x3',y3',y4',S5')) => M(x4,x3,y3,y4,S5,x4',x3',y3',y4',S5')
	M.reshape(dx4, dx3, dy3, dy4, dS5, dx4, dx3, dy3, dy4, dS5) ;
	// M(x4,x3,y3,y4,S5,x4',x3',y3',y4',S5') => M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(2, 1, 3, 4, 5, 7, 6, 8, 9, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}
// 12 -> 23 -> 47
Tensor Variation::compute_H5_47(Tensor &H47, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H47(S4,S7,S4',S7') ~~ H23(S2,S3,S2',S3')
	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5'))
	Tensor M = compute_H5_23(H47, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5')) => M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')
	M.reshape(dy3, dy4, dx3, dx4, dS5, dy3, dy4, dx3, dx4, dS5) ;
	// M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')   =>   M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(3, 4, 1, 2, 5, 8, 9, 6, 7, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}
// 12 -> 23 -> 47 -> 69
Tensor Variation::compute_H5_69(Tensor &H69, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H69(S6,S9,S6',S9') ~~ H47(S4,S7,S4',S7')
	// M((x4,x3,y3,y4,S5),(x4',x3',y3',y4',S5'))
	Tensor M = compute_H5_47(H69, wave, rt) ;

	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((x4,x3,y3,y4,S5),(x4',x3',y3',y4',S5')) => M(x4,x3,y3,y4,S5,x4',x3',y3',y4',S5')
	M.reshape(dx4, dx3, dy3, dy4, dS5, dx4, dx3, dy3, dy4, dS5) ;
	// M(x4,x3,y3,y4,S5,x4',x3',y3',y4',S5')   =>   M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(2, 1, 3, 4, 5, 7, 6, 8, 9, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}
// 12 -> 23 -> 47 -> 69 -> 89
Tensor Variation::compute_H5_89(Tensor &H89, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H89(S8,S9,S8',S9') ~~ H69(S6,S9,S6',S9')
	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5'))
	Tensor M = compute_H5_69(H89, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5')) => M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')
	M.reshape(dy3, dy4, dx3, dx4, dS5, dy3, dy4, dx3, dx4, dS5) ;
	// M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')   =>   M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(3, 4, 1, 2, 5, 8, 9, 6, 7, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}


Tensor Variation::compute_H5_25(Tensor &H25, WaveFunction &wave, ReducedTensor &rt)
{
	// A1(x1,y1,S1)
	int dx1 = wave.A1.dim[0] ;
	int dy1 = wave.A1.dim[1] ;
	int dS1 = wave.A1.dim[2] ;
	// A3(x2,y5,S3)
	int dx2 = wave.A3.dim[0] ;
	int dy5 = wave.A3.dim[1] ;
	int dS3 = wave.A3.dim[2] ;
	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;
	//----------------------------------------------------------
	// 1: T8*T9
	// M(X5,Y4,Y6) = sum{X6}_[T8(X5,X6,Y4) * T9(X6,Y6)]
	Tensor M = contractTensors(rt.T8, 2, rt.T9, 1) ;

	// 2: T7*M
	// M(Y2,Y4,Y6) = sum{X5}_[T7(X5,Y2)*M(X5,Y4,Y6)]
	M = contractTensors(rt.T7, 1, M, 1) ;

	// 3: T6*M
	// M(X4,Y5,Y2,Y4) = sum{Y6}_[T6(X4,Y5,Y6)*M(Y2,Y4,Y6)]
	M = contractTensors(rt.T6, 3, M, 3) ;

	// 4: T3*M
	// M(X2,X4,Y2,Y4) = sum{Y5}_[T3(X2,Y5)*M(X4,Y5,Y2,Y4)]
	M = contractTensors(rt.T3, 2, M, 2) ;

	// 5: T1*T4
	// m1(X1,X3,Y2) = sum{Y1}_[T1(X1,Y1)*T4(X3,Y1,Y2)]
	Tensor m1 = contractTensors(rt.T1, 2, rt.T4, 2) ;

	// 6-1: A2'*H25
	// m2(x1',x2',y3',S2,S5,S5') = sum{S2'}_[A2(x1',x2',y3',S2')*H25(S2,S5,S2',S5')]
	Tensor m2 = contractTensors(wave.A2, 4, H25, 3) ;

	// 6-2: A2*m2
	// m2(x1,x2,y3,x1',x2',y3',S5,S5') = sum{S2}_[A2(x1,x2,y3,S2)*m2(x1',x2',y3',S2,S5,S5')]
	m2 = contractTensors(wave.A2, 4, m2, 4) ;

	// m2(x1,x2,y3,x1',x2',y3',S5,S5') => m2(x1,x1',x2,x2',y3,y3',S5,S5')
	m2.permute(1, 4, 2, 5, 3, 6, 7, 8) ;

	// m2(x1,x1',x2,x2',y3,y3',S5,S5') => m2((x1,x1'),(x2,x2'),(y3,y3'),S5,S5') ~ m2(X1,X2,Y3,S5,S5')
	m2.reshape(dx1*dx1, dx2*dx2, dy3*dy3, dS5, dS5) ;

	// 6-3: m1*m2
	// m1(X3,Y2,X2,Y3,S5,S5') = sum{X1}_[m1(X1,X3,Y2)*m2(X1,X2,Y3,S5,S5')]
	m1 = contractTensors(m1, 1, m2, 1) ;

	// 7: m1*M
	// M(X3,Y3,S5,S5',X4,Y4) = sum{X2,Y2}_[m1(X3,Y2,X2,Y3,S5,S5')*M(X2,X4,Y2,Y4)]
	M = contractTensors(m1, 3, 2, M, 1, 3) ;

	// M(X3,Y3,S5,S5',X4,Y4) => M(x3,x3',y3,y3',S5,S5',x4,x4',y4,y4')
	M.reshape(dx3, dx3, dy3, dy3, dS5, dS5, dx4, dx4, dy4, dy4) ;

	// M(x3,x3',y3,y3',S5,S5',x4,x4',y4,y4') => M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(1, 7, 3, 9, 5, 2, 8, 4, 10, 6) ;

	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}
// 25 -> 45
Tensor Variation::compute_H5_45(Tensor &H45, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H45(S4,S5,S4',S5') ~~ H25(S2,S5,S2',S5')
	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5'))
	Tensor M = compute_H5_25(H45, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5')) => M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')
	M.reshape(dy3, dy4, dx3, dx4, dS5, dy3, dy4, dx3, dx4, dS5) ;
	// M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')   =>   M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(3, 4, 1, 2, 5, 8, 9, 6, 7, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}
// 25 -> 45 -> 56
Tensor Variation::compute_H5_56(Tensor &H56, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A3,A2,A1,A6,A5,A4,A9,A8,A7)
	wave.reflectAxis28() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T3,T2,T1,T6,T5,T4,T9,T8,T7)
	rt.reflectAxis28() ;
	// H56(S5,S6,S5',S6') => H56(S6,S5,S6',S5') ~~ H45(S4,S5,S4',S5')
	H56.permute(2, 1, 4, 3) ;
	// M((x4,x3,y3,y4,S5),(x4',x3',y3',y4',S5'))
	Tensor M = compute_H5_45(H56, wave, rt) ;

	// H56(S6,S5,S6',S5') => H56(S5,S6,S5',S6')
	H56.permute(2, 1, 4, 3) ;
	// (T3,T2,T1,T6,T5,T4,T9,T8,T7) => (T1,T2,T3,T4,T5,T6,T7,T8,T9)
	rt.reflectAxis28() ;
	// (A3,A2,A1,A6,A5,A4,A9,A8,A7) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis28() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((x4,x3,y3,y4,S5),(x4',x3',y3',y4',S5')) => M(x4,x3,y3,y4,S5,x4',x3',y3',y4',S5')
	M.reshape(dx4, dx3, dy3, dy4, dS5, dx4, dx3, dy3, dy4, dS5) ;
	// M(x4,x3,y3,y4,S5,x4',x3',y3',y4',S5') => M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(2, 1, 3, 4, 5, 7, 6, 8, 9, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}
// 25 -> 45 -> 56 -> 58
Tensor Variation::compute_H5_58(Tensor &H58, WaveFunction &wave, ReducedTensor &rt)
{
	// (A1,A2,A3,A4,A5,A6,A7,A8,A9) => (A1,A4,A7,A2,A5,A8,A3,A6,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;
	// H58(S5,S8,S5',S8') ~~ H56(S5,S6,S5',S6')
	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5'))
	Tensor M = compute_H5_56(H58, wave, rt) ;
	// (A1,A4,A7,A2,A5,A8,A3,A6,A9) => (A1,A2,A3,A4,A5,A6,A7,A8,A9)
	wave.reflectAxis19() ;
	// (T1,T2,T3,T4,T5,T6,T7,T8,T9) => (T1,T4,T7,T2,T5,T8,T3,T6,T9)
	rt.reflectAxis19() ;

	// A5(x3,x4,y3,y4,S5)
	int dx3 = wave.A5.dim[0] ;
	int dx4 = wave.A5.dim[1] ;
	int dy3 = wave.A5.dim[2] ;
	int dy4 = wave.A5.dim[3] ;
	int dS5 = wave.A5.dim[4] ;

	// M((y3,y4,x3,x4,S5),(y3',y4',x3',x4',S5')) => M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')
	M.reshape(dy3, dy4, dx3, dx4, dS5, dy3, dy4, dx3, dx4, dS5) ;
	// M(y3,y4,x3,x4,S5,y3',y4',x3',x4',S5')   =>   M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5')
	M.permute(3, 4, 1, 2, 5, 8, 9, 6, 7, 10) ;
	// M(x3,x4,y3,y4,S5,x3',x4',y3',y4',S5') => M((x3,x4,y3,y4,S5),(x3',x4',y3',y4',S5'))
	M.reshape(dx3*dx4*dy3*dy4*dS5, dx3*dx4*dy3*dy4*dS5) ;

	return M ;
}

//========================================================================================================
// H*A = e*N*A
void Variation::minimizeEnergyOneSite(Tensor &N, Tensor &H, Tensor &A, int latticeSize)
{
	Tensor E = symGenEig(H, N, A, 1) ;
	energy = E(1) / latticeSize ; // energy per site
	//energy = E(1) ;
	//cout << "energy = " << energy << endl ;
	//cout << "energy error = " << energy + 1.0 / 3.0 << endl ;
	//cout << "energy error = " << energy + 0.527703028728097 << endl ;
	//*********************************** debug *****************************************
	/*
	int n = prod(A.dim) ;
	Tensor V = Tensor(n) ; // column vector
	V.value = A.value ;

	Tensor X = H * V ;

	Tensor Vt = V ;
	Vt.trans() ; // Vt = V'

	X = Vt * X ;

	Tensor Y = N * V ;
	Y = Vt * Y ;

	double e = X(1) / Y(1) ;
	e = e / latticeSize ;
	cout << "e = " << e << endl ;
	*/
 	//***********************************************************************************

}
//=============================================================================================
Variation::~Variation() {}
