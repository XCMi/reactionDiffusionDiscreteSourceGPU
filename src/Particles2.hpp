/*
  Copyright © Cambridge Numerical Solutions Ltd 2013
*/
#pragma once
#include "Cloud2.hpp"

template<typename T, Processor processor, int Length>
class Particles2D : public Cloud2D<T, processor, Length> {
protected:
        const real NP_;
	real time_;

	typedef Cloud2D<T, processor, Length> Parent;

public:
	typedef T Type;

	Particles2D(const int NP) :
		Parent(NP),
                NP_(NP),
		time_(0.0) {
	}

	// copy constructor: deep, defined over different processor types.
	template<Processor sourceProcessor>
	Particles2D(const Particles2D<T, sourceProcessor, Length>& copy, const typename Parent::CopyType copyType = Parent::DeepCopy) :
		Parent(copy, copyType),
                NP_(copy.NP_),
		time_(copy.time_) {
	}

	Particles2D(const Particles2D<T, processor, Length>& copy, const typename Parent::CopyType copyType = Parent::ShallowCopy) :
		Parent(copy, copyType),
                NP_(copy.NP_),
		time_(copy.time_) {
	}

	// assignment: deep copy. defined either way.
	template<Processor sourceProcessor>
	Particles2D<T, processor, Length>& operator=(const Particles2D<T, sourceProcessor, Length>& copy) {
		time_ = copy.time_;
		Parent::operator=(copy);
		return *this;
	}

	// assignment over the same type
	Particles2D& operator=(const Particles2D<T, processor, Length>& copy) {
		time_ = copy.time_;
		Parent::operator=(copy);
		return *this;
	}
        /*
	__host__ __device__  StridedCell<T, Length> operator()(const int x, const int y) const {
		return StridedCell<T, Length>(Parent::grid + (x + ghostCells()) + Parent::Nx_ * (y + ghostCells()), Parent::Nx_ * Parent::Ny_);
	}
        */
	__host__ __device__ T& operator()(const int pi, const int variable) {
		return Parent::operator()(pi, variable);
	}
	
	__host__ __device__ T operator()(const int pi, const int variable) const {
		return Parent::operator()(pi, variable);
	}

	// TRIVIAL UTILITY FUNCTIONS
	__host__ __device__ real particle_index(const int pi) const { return 1.0 * pi; }
        __host__ __device__ bool active(const int i) const { return i >= 0 && i < NP(); }
        /*
	__host__ __device__ int activeNx() const { return Parent::Nx() - 2 * ghostCells(); }
	__host__ __device__ int activeNy() const { return Parent::Ny() - 2 * ghostCells(); }
	__host__ __device__ Vec d() const { return Vec(dx(), dy()); }
	__host__ __device__ real dx() const { return (xMax_ - xMin_) / activeNx(); }
	__host__ __device__ real dy() const { return (yMax_ - yMin_) / activeNy(); }
	__host__ __device__ int ghostCells() const { return ghostCells_; }
	__host__ __device__ real x(const int i) const { return xMin_ + i * dx() + dx() / 2.0; }
	__host__ __device__ real y(const int j) const { return yMin_ + j * dy() + dy() / 2.0; }
	__host__ __device__ Vec x(const int i, const int j) { return Vec(x(i), y(j)); }
	__host__ __device__ real i(const real x) const { return (x - xMin_) / dx() - 0.5; }
	__host__ __device__ real j(const real y) const { return (y - yMin_) / dy() - 0.5; }

	__host__ __device__ bool active(const int i, const int j) const { return i >= 0 && i < activeNx() && j >= 0 && j < activeNy(); }
	__host__ __device__ bool within(const int i, const int j, const int band) const { return i >= -ghostCells() + band && i < activeNx() + ghostCells() - band && j >= -ghostCells() + band && j < activeNy() + ghostCells() - band; }
	__host__ __device__ bool exists(const int i, const int j) const { return i >= -ghostCells() && i < activeNx() + ghostCells() && j >= -ghostCells() && j < activeNy() + ghostCells(); }
	__host__ __device__ dim3 activeCover(const dim3 blockSize, const int overlapX = 0, const int overlapY = 0) const {
		return this->cover(blockSize, 0, Parent::Nx() - 2 * ghostCells(), 0, Parent::Ny() - 2 * ghostCells(), overlapX, overlapY);
	}
        */

	// GETTERS AND SETTERS
        /*
	__host__ __device__ real xMin() const { return xMin_; }
	__host__ __device__ real yMin() const { return yMin_; }
	__host__ __device__ real xMax() const { return xMax_; }
	__host__ __device__ real yMax() const { return yMax_; }
        */
        __host__ __device__ real NP() const { return NP_; }
	__host__ __device__ real time() const { return time_; }
	__host__ __device__ void time(const real& time) { time_ = time; }

	template<typename, Processor, int>
	friend class Particles2D;
};

