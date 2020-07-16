/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#pragma once
#include "Cloud3.hpp"

template<typename T, Processor processor, int Length>
class Particles3D : public Cloud3D<T, processor, Length> {
protected:
        const real NP_;
	real time_;

	typedef Cloud3D<T, processor, Length> Parent;

public:
	typedef T Type;

	Particles3D(const int NP) :
		Parent(NP),
                NP_(NP),
		time_(0.0) {
	}

	// copy constructor: deep, defined over different processor types.
	template<Processor sourceProcessor>
	Particles3D(const Particles3D<T, sourceProcessor, Length>& copy, const typename Parent::CopyType copyType = Parent::DeepCopy) :
		Parent(copy, copyType),
                NP_(copy.NP_),
		time_(copy.time_) {
	}

	Particles3D(const Particles3D<T, processor, Length>& copy, const typename Parent::CopyType copyType = Parent::ShallowCopy) :
		Parent(copy, copyType),
                NP_(copy.NP_),
		time_(copy.time_) {
	}

	// assignment: deep copy. defined either way.
	template<Processor sourceProcessor>
	Particles3D<T, processor, Length>& operator=(const Particles3D<T, sourceProcessor, Length>& copy) {
		time_ = copy.time_;
		Parent::operator=(copy);
		return *this;
	}

	// assignment over the same type
	Particles3D& operator=(const Particles3D<T, processor, Length>& copy) {
		time_ = copy.time_;
		Parent::operator=(copy);
		return *this;
	}

	__host__ __device__ T& operator()(const int pi, const int variable) {
		return Parent::operator()(pi, variable);
	}
	
	__host__ __device__ T operator()(const int pi, const int variable) const {
		return Parent::operator()(pi, variable);
	}

	// TRIVIAL UTILITY FUNCTIONS
	__host__ __device__ real particle_index(const int pi) const { return 1.0 * pi; }
    __host__ __device__ bool active(const int i) const { return i >= 0 && i < NP(); }
    __host__ __device__ real NP() const { return NP_; }
	__host__ __device__ real time() const { return time_; }
	__host__ __device__ void time(const real& time) { time_ = time; }

	template<typename, Processor, int>
	friend class Particles3D;
};

