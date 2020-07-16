/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#pragma once
#include "Grid3.hpp"

template<typename T, Processor processor, int Length>
class Mesh3D : public Grid3D<T, processor, Length> {
protected:
	const real xMin_;
	const real xMax_;
	const real yMin_;
	const real yMax_;
	const real zMin_;
	const real zMax_;
	int ghostCells_;
	real time_;

	typedef Grid3D<T, processor, Length> Parent;

public:
	typedef T Type;

	Mesh3D(const int Nx, const int Ny, const int Nz, const int ghostCells, const real xMin = 0.0, const real xMax = 1.0, const real yMin = 0.0, const real yMax = 1.0, const real zMin = 0.0, const real zMax = 1.0) :
		Parent(Nx + 2 * ghostCells, Ny + 2 * ghostCells, Nz + 2 * ghostCells),
		ghostCells_(ghostCells),
		xMin_(xMin),
		xMax_(xMax),
		yMin_(yMin),
		yMax_(yMax),
		zMin_(zMin),
		zMax_(zMax),
		time_(0.0) {
	}

	// copy constructor: deep, defined over different processor types.
	template<Processor sourceProcessor>
	Mesh3D(const Mesh3D<T, sourceProcessor, Length>& copy, const typename Parent::CopyType copyType = Parent::DeepCopy) :
		Parent(copy, copyType),
		ghostCells_(copy.ghostCells_),
		xMin_(copy.xMin_),
		xMax_(copy.xMax_),
		yMin_(copy.yMin_),
		yMax_(copy.yMax_),
		zMin_(copy.zMin_),
		zMax_(copy.zMax_),
		time_(copy.time_) {
	}

	Mesh3D(const Mesh3D<T, processor, Length>& copy, const typename Parent::CopyType copyType = Parent::ShallowCopy) :
		Parent(copy, copyType),
		ghostCells_(copy.ghostCells_),
		xMin_(copy.xMin_),
		xMax_(copy.xMax_),
		yMin_(copy.yMin_),
		yMax_(copy.yMax_),
		zMin_(copy.zMin_),
		zMax_(copy.zMax_),
		time_(copy.time_) {
	}

	// assignment: deep copy. defined either way.
	template<Processor sourceProcessor>
	Mesh3D<T, processor, Length>& operator=(const Mesh3D<T, sourceProcessor, Length>& copy) {
		time_ = copy.time_;
		Parent::operator=(copy);
		return *this;
	}

	// assignment over the same type
	Mesh3D& operator=(const Mesh3D<T, processor, Length>& copy) {
		time_ = copy.time_;
		Parent::operator=(copy);
		return *this;
	}

	__host__ __device__  StridedCell<T, Length> operator()(const int x, const int y, const int z) const {
		return StridedCell<T, Length>(Parent::grid + (x + ghostCells()) + Parent::Nx_ * (y + ghostCells() + Parent::Ny_ * (z + ghostCells())), Parent::Nx_ * Parent::Ny_ * Parent::Nz_);
	}

	__host__ __device__ T& operator()(const int x, const int y, const int z, const int variable) {
		return Parent::operator()(x + ghostCells(), y + ghostCells(), z + ghostCells(), variable);
	}
	
	__host__ __device__ T operator()(const int x, const int y, const int z, const int variable) const {
		return Parent::operator()(x + ghostCells(), y + ghostCells(), z + ghostCells(), variable);
	}

	// TRIVIAL UTILITY FUNCTIONS
	__host__ __device__ int activeNx() const { return Parent::Nx() - 2 * ghostCells(); }
	__host__ __device__ int activeNy() const { return Parent::Ny() - 2 * ghostCells(); }
	__host__ __device__ int activeNz() const { return Parent::Nz() - 2 * ghostCells(); }
	__host__ __device__ Vec d() const { return Vec(dx(), dy(), dz()); }
	__host__ __device__ real dx() const { return (xMax_ - xMin_) / activeNx(); }
	__host__ __device__ real dy() const { return (yMax_ - yMin_) / activeNy(); }
	__host__ __device__ real dz() const { return (zMax_ - zMin_) / activeNz(); }
	__host__ __device__ int ghostCells() const { return ghostCells_; }
	__host__ __device__ real x(const int i) const { return xMin_ + i * dx() + dx() / 2.0; }
	__host__ __device__ real y(const int j) const { return yMin_ + j * dy() + dy() / 2.0; }
	__host__ __device__ real z(const int k) const { return zMin_ + k * dz() + dz() / 2.0; }
	__host__ __device__ Vec x(const int i, const int j, const int k) { return Vec(x(i), y(j), z(k)); }
	__host__ __device__ real i(const real x) const { return (x - xMin_) / dx() - 0.5; }
	__host__ __device__ real j(const real y) const { return (y - yMin_) / dy() - 0.5; }
	__host__ __device__ real k(const real z) const { return (z - zMin_) / dz() - 0.5; }

	__host__ __device__ bool active(const int i, const int j, const int k) const { return i >= 0 && i < activeNx() && j >= 0 && j < activeNy() && k >= 0 && k < activeNz(); }
	__host__ __device__ bool within(const int i, const int j, const int k, const int band) const { return i >= -ghostCells() + band && i < activeNx() + ghostCells() - band && j >= -ghostCells() + band && j < activeNy() + ghostCells() - band && k >= -ghostCells() + band && k < activeNz() + ghostCells() - band; }
	__host__ __device__ bool exists(const int i, const int j, const int k) const { return i >= -ghostCells() && i < activeNx() + ghostCells() && j >= -ghostCells() && j < activeNy() + ghostCells() && k >= -ghostCells() && k < activeNz() + ghostCells(); }
	__host__ __device__ dim3 activeCover(const dim3 blockSize, const int overlapX = 0, const int overlapY = 0, const int overlapZ = 0) const {
		return this->cover(blockSize, 0, Parent::Nx() - 2 * ghostCells(), 0, Parent::Ny() - 2 * ghostCells(), 0, Parent::Nz() - 2 * ghostCells(), overlapX, overlapY, overlapZ);
	}

	// GETTERS AND SETTERS
	__host__ __device__ real xMin() const { return xMin_; }
	__host__ __device__ real yMin() const { return yMin_; }
	__host__ __device__ real zMin() const { return zMin_; }
	__host__ __device__ real xMax() const { return xMax_; }
	__host__ __device__ real yMax() const { return yMax_; }
	__host__ __device__ real zMax() const { return zMax_; }
	__host__ __device__ real time() const { return time_; }
	__host__ __device__ void time(const real& time) { time_ = time; }

	template<typename, Processor, int>
	friend class Mesh3D;
};

