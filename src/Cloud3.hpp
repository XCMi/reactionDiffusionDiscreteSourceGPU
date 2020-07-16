/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#pragma once
#include <iostream>

// enum Processor {CPU, GPU};

template<typename T, Processor processor, int Length = 1>
class Cloud3D {
protected:
  T* cloud;
  const int NP_;
  const int length_;

  void allocateCloud();

public:
  enum CopyType {
    DeepCopy,    // makes a copy of cloud data
    ShallowCopy, // copies a pointer to cloud data
    Allocate     // just allocates space, but copies other parameters
  };

  Cloud3D(const int NP) :
    NP_(NP),
    length_(Length) {
    allocateCloud();
  }

  // copy constructor: deep, defined over different processor types.
  template<Processor sourceProcessor>
  Cloud3D(const Cloud3D<T, sourceProcessor, Length>& copy, const CopyType copyType = DeepCopy);

  Cloud3D(const Cloud3D<T, processor, Length>& copy, const CopyType copyType = ShallowCopy);

  cudaMemcpyKind getMemcpyKind(const Processor sourceProcessor, const Processor destinationProcessor);

  // assignment: deep copy. defined either way.
  template<Processor sourceProcessor>
  Cloud3D<T, processor, Length>& operator=(const Cloud3D<T, sourceProcessor, Length>& copy);

  // assignment over the same type
  Cloud3D& operator=(const Cloud3D<T, processor, Length>& copy);

  ~Cloud3D() { }

  __host__ __device__ __forceinline__ T& operator()(const int pi, const int variable) {
    return cloud[pi + NP_ * variable];
  }
  __host__ __device__ __forceinline__ T operator()(const int pi, const int variable) const {
    return cloud[pi + NP_ * variable];
  }

  __host__ __device__ __forceinline__ T* getCloud() { return cloud; }

  // GETTERS AND SETTERS
  __host__ __device__ __forceinline__ int NP() const { return NP_; }
  __host__ __device__ __forceinline__ int length() const { return Length; }

  // TRIVIAL UTILITY FUNCTIONS
  __host__ __device__ __forceinline__ bool contains(const int i) const { return i >= 0 && i < NP_; }

  __host__ __device__ dim3 cover(const dim3 blockSize, const int iStart, const int iEnd) const;

  __host__ __device__ dim3 totalCover(const dim3 blockSize) const;

  void free();

  template<typename S, Processor P, int L>
  friend class Cloud3D;

  friend void swap(Cloud3D<T, processor, Length>& a, Cloud3D<T, processor, Length>& b) {
    using std::swap;
    swap(a.cloud, b.cloud);
  }
};


template<typename T, Processor processor, int Length>
void
Cloud3D<T, processor, Length>::
allocateCloud() {
  switch (processor) {
    case CPU:
      cloud = new T[NP_ * Length];
      break;
    case GPU:
      cudaMalloc(&cloud, sizeof(T) * NP_ * Length);
      checkForError();
      break;
  }
}

template<typename T, Processor processor, int Length>
template<Processor sourceProcessor>
Cloud3D<T, processor, Length>::
Cloud3D(const Cloud3D<T, sourceProcessor, Length>& copy, const CopyType copyType) :
  NP_(copy.NP_),
  length_(Length) {
  // use the assignment copy to deep copy non-const data
  allocateCloud();
  switch (copyType) {
    case Allocate:
    break;

    default:
      *this = copy;
    break;
  }
}

template<typename T, Processor processor, int Length>
Cloud3D<T, processor, Length>::
Cloud3D(const Cloud3D<T, processor, Length>& copy, const CopyType copyType) :
  NP_(copy.NP_),
  length_(Length) {
  switch (copyType) {
    case DeepCopy:
      allocateCloud();
      *this = copy;
    break;

    case Allocate:
      allocateCloud();
    break;

    default:
      cloud = copy.cloud;
    break;
  }
}

template<typename T, Processor processor, int Length>
cudaMemcpyKind
Cloud3D<T, processor, Length>::
getMemcpyKind(const Processor sourceProcessor, const Processor destinationProcessor) {
  cudaMemcpyKind copyType;
  if (destinationProcessor == CPU && sourceProcessor == CPU) {
    copyType = cudaMemcpyHostToHost;
  } else if (destinationProcessor == CPU && sourceProcessor == GPU) {
    copyType = cudaMemcpyDeviceToHost;
  } else if (destinationProcessor == GPU && sourceProcessor == CPU) {
    copyType = cudaMemcpyHostToDevice;
  } else if (destinationProcessor == GPU && sourceProcessor == GPU) {
    copyType = cudaMemcpyDeviceToDevice;
  }
  return copyType;
}

template<typename T, Processor processor, int Length>
template<Processor sourceProcessor>
Cloud3D<T, processor, Length>&
Cloud3D<T, processor, Length>::
operator=(const Cloud3D<T, sourceProcessor, Length>& copy) {
  cudaMemcpyKind copyType = getMemcpyKind(sourceProcessor, processor);

  cudaMemcpy(cloud, copy.cloud, sizeof(T) * NP_ * Length, copyType);
	checkForError();

  return *this;
}

template<typename T, Processor processor, int Length>
Cloud3D<T, processor, Length>&
Cloud3D<T, processor, Length>::
operator=(const Cloud3D<T, processor, Length>& copy) {
  cudaMemcpyKind copyType;
  if (processor == CPU) {
    copyType = cudaMemcpyHostToHost;
  } else if (processor == GPU) {
    copyType = cudaMemcpyDeviceToDevice;
  }

  cudaMemcpy(cloud, copy.cloud, sizeof(T) * NP_ * Length, copyType);

  return *this;
}

template<typename T, Processor processor, int Length>
__host__ __device__ dim3
Cloud3D<T, processor, Length>::
cover(const dim3 blockSize, const int iStart, const int iEnd) const {
  dim3 gridSize;
  gridSize.x = 1 + ( iEnd - iStart - 1) / blockSize.x;
  gridSize.y = 1;
  gridSize.z = 1;
  return gridSize;
}

template<typename T, Processor processor, int Length>
__host__ __device__ dim3
Cloud3D<T, processor, Length>::
totalCover(const dim3 blockSize) const {
  return cover(blockSize, 0, NP_);
}

/*
template<typename T, Processor processor, int Length>
__host__ __device__ dim3
Grid3D<T, processor, Length>::
excludeCover(const dim3 blockSize, const int band, const int overlapX, const int overlapY) const {
  return cover(blockSize, band, Nx_ - band, band, Ny_ - band, overlapX, overlapY);
}
*/

template<typename T, Processor processor, int Length>
void
Cloud3D<T, processor, Length>::
free() {
  switch (processor) {
    case GPU: cudaFree(this->grid); break;
    case CPU: delete[] this->grid; break;
  }
}

