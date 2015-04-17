#ifndef GPU_LOOKUP_H
#define GPU_LOOKUP_H

#ifndef _WINDOWS_
#define _WINDOWS_
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct GpuLookupRow
{
	double* row;		///< Pointer to the first column on a row
	double fraction;	///< Fraction of V or Ca over and above the division
						///< boundary for interpolation.
};

struct GpuLookupColumn
{
	unsigned int column;
};

class GpuLookupTable
{
private:
	double min_, max_, dx_;
	unsigned int nPts_, nColumns_;
	bool deviceUpdateNeeded_;
	// NOTE: SINGLE ELEMENT KERNEL ARGUMENTS WHEN PASSED BY VALUE, ARE ACTUALLY PASSED IN SHARED MEMORY
	// SO COPY THEM TO GLOBAL DEVICE MEMORY RESULTS IN A UNNECESSARY OVERHEAD IN CODE LENGTH AND COMPUTATION TIME
	// THUS THERE IS NO NEED FOR THE DEVICE VARIABLES DECLARED PREVIOUSLY HERE

	// NOTE: GATHER COLUMNS IN HOST MEMORY AND COPY TO DEVICE ONLY WHEN NEEDED
	thrust::host_vector<double> table;
	thrust::device_vector<double> table_d;

public:
	GpuLookupTable();
	GpuLookupTable(double min, double max, int nDivs, unsigned int nSpecies);	// NOTE: NO GAIN IN PASSING ARGUMENTS AS POINTERS
	void addColumns(int species, double *C1, double *C2);
	void lookup(double *row, double *column, double *istate, double dt, unsigned int set_size);
};

void calcGridBlockDims(dim3& grid, dim3& block, unsigned int size);

#endif
