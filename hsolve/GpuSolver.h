#ifndef EXAMPLE6_H
#define EXAMPLE6_H

#ifndef _WINDOWS_
#define _WINDOWS_
#endif

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

class GpuInterface
{
private:
	thrust::device_vector<double> A_d;
	thrust::device_vector<double> B_d;
	int Asize, Bsize;
	double xmin, xmax;
	double invDx;

public:
	GpuInterface();
	void setupTables(const double *A, const double *B, int Asize, int Bsize, double xmin, double xmax, double invDx);
	void lookupTables(const std::vector<double>& V, std::vector<double>& A, std::vector<double>& B) const;
};

void calcGridBlockDims(dim3& grid, dim3& block, unsigned int size);

#endif
