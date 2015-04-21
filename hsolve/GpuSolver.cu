#include <iostream>
#include <cuda.h>
#include "GpuSolver.h"

// NOTE: obviously I would not want to keep these _WINDOWS_ stuff in code. I just compile code on windows at the moment
#ifdef _WINDOWS_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#endif

using namespace std;

// NOTE: THIS IMPLEMENTATION IS FAR FROM THE OPTIMAL SOLUTION, IT NEEDS REVISION AS I MENTIONED IN MY PROPOSAL
__global__
void lookupTable(const double *V, const double *A_d, const double *B_d,
				double xmin, double xmax, double invDx, int Vsize, int Asize, int Bsize,
				double *A_out, double *B_out)
{
	const unsigned int tId = threadIdx.x + (blockDim.x * (blockIdx.x + (gridDim.x * blockIdx.y))); // 2D grid and 1D block
	if (tId >= Vsize) return;

	if (V[tId] <= xmin)
	{
		A_out[tId] = A_d[0];
		B_out[tId] = B_d[0];
	}
	else if (V[tId] >= xmax)
	{
		A_out[tId] = A_d[Asize];
		B_out[tId] = B_d[Bsize];
	}
	else
	{	
		unsigned int index = (V[tId] - xmin) * invDx;						// INDEXED BY VOLTAGE
		// check for lookupByInterpolation in the HHGate code
		double frac = (V[tId] - xmin - (index / invDx)) * invDx;			// DIFFERENCE BETWEEN VOLTAGE AND LOOKUP VOLTAGE
		A_out[tId] = A_d[index] * (1 - frac) + A_d[index+1] * frac;			// LINEAR INTERPOLATION USING THE DIFF AND INDEX
		B_out[tId] = B_d[index] * (1 - frac) + B_d[index+1] * frac;
	}
}

GpuInterface::GpuInterface() {}

//This is the reimplementation of the lookuptable code in biophysics/HHGate
//Not the one in hsolve
// NOTE: interface is changed to be able to process multiple voltage values at the same time
void GpuInterface::lookupTables(const std::vector<double>& V, std::vector<double>& A, std::vector<double>& B) const
{
	thrust::device_vector<double> V_d(V);
	thrust::device_vector<double> A_out(V.size());
	thrust::device_vector<double> B_out(V.size());

	dim3 grid, block;
	calcGridBlockDims(grid, block, V.size()); // more sophisticated grid and block dim calc is needed that takes into account the channel types
	lookupTable<<<grid, block>>>(thrust::raw_pointer_cast(V_d.data()),
		thrust::raw_pointer_cast(A_d.data()), thrust::raw_pointer_cast(B_d.data()),
		xmin, xmax, invDx, V.size(), Asize, Bsize,
		thrust::raw_pointer_cast(A_out.data()), thrust::raw_pointer_cast(B_out.data()));

	// copy back results to host
	A.assign(A_out.begin(), A_out.end());
	B.assign(B_out.begin(), B_out.end());
}


//This is the reimplementation of the setuptable code in biophysics/HHGate
//Not the one in hsolve
void GpuInterface::setupTables(const double *A, const double *B, int Asize, int Bsize, double xmin, double xmax, double invDx)
{
	this->Asize = Asize;
	this->Bsize = Bsize;
	this->xmin = xmin;
	this->xmax = xmax;
	this->invDx = invDx;

	A_d.assign(A, A + Asize);
	B_d.assign(B, B + Bsize);
}
