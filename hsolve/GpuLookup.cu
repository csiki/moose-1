#include "GpuLookup.h"
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef _WINDOWS_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

// NOTE: THE STRUCTURE OF TABLE IS UNSUITABLE FOR GPGPU CALCULATIONS
// MEMORY ACCESS PATTERN DOES NOT ALLOW USE OF TEXTURE OR CONSTANT MEMORY
// TABLE HAS TO BE STORED IN A 2D TEXTURE MEMORY - ALSO THE DEPENDENCY AMONG ROW_ARRAY, COLUMN_ARRAY AND TABLE ARE JUST TOO HIGH
__global__ void lookup_kernel(double *row_array, double *column_array, double *table, unsigned int nRows, unsigned int nColumns, double *istate, double dt)
{
	const unsigned int tId = threadIdx.x + (blockDim.x * (blockIdx.x + (gridDim.x * blockIdx.y))); // 2D grid and 1D block
	if (tId >= nRows) return;

	int row = row_array[tId];
	double fraction = row_array[tId]-row;

	double column = column_array[tId];

	double *a = table, *b = table;

	a += (int)(row + column * nRows);
	b += (int)(row + column * nRows + nRows);

	double C1 = *a + (*(a+1) - *a) * fraction;
	double C2 = *b + (*(b+1) - *b) * fraction;

	double temp = 1.0 + dt / 2.0 * C2;
    istate[tId] = ( istate[tId] * ( 2.0 - temp ) + dt * C1 ) / temp;
}

GpuLookupTable::GpuLookupTable() {}

GpuLookupTable::GpuLookupTable(double min, double max, int nDivs, unsigned int nSpecies)
	: min_(min), max_(max), deviceUpdateNeeded_(false)
{
	// Number of points is 1 more than number of divisions.
	// Then add one more since we may interpolate at the last point in the table.
	nPts_ = nDivs + 1 + 1;
	dx_= ( max - min ) / nDivs;
	// Every row has 2 entries for each type of gate
	nColumns_ = 0; //2 * nSpecies;
}

// Columns are arranged in memory as    |	They are visible as
// 										|	
//										|	Column 1 	Column 2 	Column 3 	Column 4
// C1(Type 1)							|	C1(Type 1) 	C2(Type 1)	C1(Type 2)	C2(Type 2)
// C2(Type 1)							|	
// C1(Type 2)							|
// C2(Type 2)							|
// .									|
// .									|
// .									|


void GpuLookupTable::addColumns(int species, double *C1, double *C2)
{
	// NOTE: COPYING COLUMN VALUES TO GLOBAL DEVICE MEMORY ONE BY ONE IS EXTREMELY SLOW AND UNNECESSARY
	// HOST_VECTOR CAN BE POPULATED WITHOUT ANY MEMORY TRAFFIC OVERHEAD AND ONLY COPIED TO DEVICE WHEN GPU CALCULATION IS INITIATED
	nColumns_ += 2;
	table.reserve(nPts_ * (nColumns_));

	// add C1 values
	for (size_t i = 0; i < nPts_ - 1; ++i)
		table.push_back(C1[i]);
	table.push_back(C1[nPts_ - 2]);

	// add C2 values
	for (size_t i = 0; i < nPts_ - 1; ++i)
		table.push_back(C2[i]);
	table.push_back(C2[nPts_ - 2]);

	deviceUpdateNeeded_ = true;
}

void GpuLookupTable::lookup(double *row, double *column, double *istate, double dt, unsigned int set_size)
{
	// NOTE: THESE DO THE ALLOCATION OF DEVICE MEMORY AND ALSO THE COPY FROM HOST TO DEVICE IN ONE STATEMENT
	// WITHOUT ANY OVERHEAD COMPARED TO THE ORIGINAL SOLUTION
	thrust::device_vector<double> row_d(row, row + set_size);
	thrust::device_vector<double> col_d(column, column + set_size);
	thrust::device_vector<double> istate_d(istate, istate + set_size);
	if (deviceUpdateNeeded_)
	{
		thrust::device_vector<double> table_d = table; // here complete table is copied to device memory
		deviceUpdateNeeded_ = false;
	}

	dim3 grid, block;
	calcGridBlockDims(grid, block, set_size);
	lookup_kernel<<<grid, block>>>(thrust::raw_pointer_cast(row_d.data()), thrust::raw_pointer_cast(col_d.data()),
		thrust::raw_pointer_cast(table_d.data()), nPts_, nColumns_, thrust::raw_pointer_cast(istate_d.data()), dt);

	// retrieve calculated states (copy back from device)
	thrust::copy(istate_d.begin(), istate_d.end(), istate);
}

// NOTE: MACHINES WITH MULTIPLE DEVICES HAS TO BE TAKEN INTO ACCOUNT AS WELL - DECIDE ON WHICH LEVEL THE COMPUTATION IS DIVIDED AMONG DEVICES
// The kernel parameters can be defined automatically and optimally if the kernel has to calculate only along a 1D vector
void calcGridBlockDims(dim3& grid, dim3& block, unsigned int size)
{
	unsigned int threadNum = ((size - 1) / 32 + 1) * 32; // multiple of 32
	if (threadNum > 512) // 512 or 1024 is the maximum number of threads
		threadNum = 512;
	block = dim3(threadNum);
	
	grid = dim3();
	grid.x = (size - 1) / threadNum + 1;
	if (grid.x > 65535) // 65535 is the max number of blocks in one dimension
	{
		grid.x = sqrt(grid.x - 1) + 1;
		grid.y = grid.x;
	}
}