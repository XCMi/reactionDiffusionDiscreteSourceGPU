/*
  Copyright © Computing Energetic - Calcul Energetique 2020
*/
#pragma once
#include <iostream>

enum Processor {CPU, GPU};

template<typename T, Processor processor, int Length = 1>
class Grid3D {
protected:
  T* grid;
  const int Nx_;
  const int Ny_;
  const int Nz_;
  const int length_;

  void allocateGrid();

public:
  enum CopyType {
    DeepCopy,    // makes a copy of grid data
    ShallowCopy, // copies a pointer to grid data
    Allocate     // just allocates space, but copies other parameters
  };

  Grid3D(const int Nx, const int Ny, const int Nz) :
    Nx_(Nx),
    Ny_(Ny),
    Nz_(Nz),
    length_(Length) {
    allocateGrid();
  }

  // copy constructor: deep, defined over different processor types.
  template<Processor sourceProcessor>
  Grid3D(const Grid3D<T, sourceProcessor, Length>& copy, const CopyType copyType = DeepCopy);

  Grid3D(const Grid3D<T, processor, Length>& copy, const CopyType copyType = ShallowCopy);

  cudaMemcpyKind getMemcpyKind(const Processor sourceProcessor, const Processor destinationProcessor);

  // assignment: deep copy. defined either way.
  template<Processor sourceProcessor>
  Grid3D<T, processor, Length>& operator=(const Grid3D<T, sourceProcessor, Length>& copy);

  // assignment over the same type
  Grid3D& operator=(const Grid3D<T, processor, Length>& copy);

  ~Grid3D() { }

  __host__ __device__ __forceinline__ T& operator()(const int x, const int y, const int z, const int variable) {
    return grid[x + Nx_ * (y + Ny_ * (z + Nz_ * variable))];
  }
  __host__ __device__ __forceinline__ T operator()(const int x, const int y, const int z, const int variable) const {
    return grid[x + Nx_ * (y + Ny_ * (z + Nz_ * variable))];
  }

  __host__ __device__ __forceinline__ T* getGrid() { return grid; }

  // GETTERS AND SETTERS
  __host__ __device__ __forceinline__ int Nx() const { return Nx_; }
  __host__ __device__ __forceinline__ int Ny() const { return Ny_; }
  __host__ __device__ __forceinline__ int Nz() const { return Nz_; }
  __host__ __device__ __forceinline__ int length() const { return Length; }

  // TRIVIAL UTILITY FUNCTIONS
  __host__ __device__ __forceinline__ bool contains(const int i, const int j, const int k) const { return i >= 0 && i < Nx_ && j >= 0 && j < Ny_ && k >= 0 && k < Nz_; }

  __host__ __device__ dim3 cover(const dim3 blockSize, const int iStart, const int iEnd, const int jStart, const int jEnd, const int kStart, const int kEnd, const int overlapX = 0, const int overlapY = 0, const int overlapZ = 0) const;

  __host__ __device__ dim3 totalCover(const dim3 blockSize, const int overlapX = 0, const int overlapY = 0, const int overlapZ = 0) const;

  __host__ __device__ dim3 excludeCover(const dim3 blockSize, const int band = 0, const int overlapX = 0, const int overlapY = 0, const int overlapZ = 0) const;

  __host__ __device__ __forceinline__ real* getAddress(const int x, const int y, const int z, const int variable) {
    return &grid[x + Nx_ * (y + Ny_ * (z + Nz_ * variable))];
  }

  void free();

  template<typename S, Processor P, int L>
  friend class Grid3D;

  friend void swap(Grid3D<T, processor, Length>& a, Grid3D<T, processor, Length>& b) {
    using std::swap;
    swap(a.grid, b.grid);
  }
};


template<typename T, Processor processor, int Length>
void
Grid3D<T, processor, Length>::
allocateGrid() {
  switch (processor) {
    case CPU:
      grid = new T[Nx_ * Ny_ * Nz_ * Length];
      break;
    case GPU:
      cudaMalloc(&grid, sizeof(T) * Nx_ * Ny_ * Nz_ * Length);
      checkForError();
      break;
  }
}

template<typename T, Processor processor, int Length>
template<Processor sourceProcessor>
Grid3D<T, processor, Length>::
Grid3D(const Grid3D<T, sourceProcessor, Length>& copy, const CopyType copyType) :
  Nx_(copy.Nx_),
  Ny_(copy.Ny_),
  Nz_(copy.Nz_),
  length_(Length) {
  // use the assignment copy to deep copy non-const data
  allocateGrid();
  switch (copyType) {
    case Allocate:
    break;

    default:
      *this = copy;
    break;
  }
}

template<typename T, Processor processor, int Length>
Grid3D<T, processor, Length>::
Grid3D(const Grid3D<T, processor, Length>& copy, const CopyType copyType) :
  Nx_(copy.Nx_),
  Ny_(copy.Ny_),
  Nz_(copy.Nz_),
  length_(Length) {
  switch (copyType) {
    case DeepCopy:
      allocateGrid();
      *this = copy;
    break;

    case Allocate:
      allocateGrid();
    break;

    default:
      grid = copy.grid;
    break;
  }
}

template<typename T, Processor processor, int Length>
cudaMemcpyKind
Grid3D<T, processor, Length>::
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
Grid3D<T, processor, Length>&
Grid3D<T, processor, Length>::
operator=(const Grid3D<T, sourceProcessor, Length>& copy) {
  cudaMemcpyKind copyType = getMemcpyKind(sourceProcessor, processor);

  cudaMemcpy(grid, copy.grid, sizeof(T) * Nx_ * Ny_ * Nz_ * Length, copyType);
	checkForError();

  return *this;
}

template<typename T, Processor processor, int Length>
Grid3D<T, processor, Length>&
Grid3D<T, processor, Length>::
operator=(const Grid3D<T, processor, Length>& copy) {
  cudaMemcpyKind copyType;
  if (processor == CPU) {
    copyType = cudaMemcpyHostToHost;
  } else if (processor == GPU) {
    copyType = cudaMemcpyDeviceToDevice;
  }

  cudaMemcpy(grid, copy.grid, sizeof(T) * Nx_ * Ny_ * Nz_ * Length, copyType);

  return *this;
}

template<typename T, Processor processor, int Length>
__host__ __device__ dim3
Grid3D<T, processor, Length>::
cover(const dim3 blockSize, const int iStart, const int iEnd, const int jStart, const int jEnd, const int kStart, const int kEnd, const int overlapX, const int overlapY, const int overlapZ) const {
  dim3 gridSize;
  gridSize.x = 1 + ( iEnd - iStart - 1 * overlapX - 1) / (blockSize.x - 1 * overlapX);
  gridSize.y = 1 + ( jEnd - jStart - 1 * overlapY - 1) / (blockSize.y - 1 * overlapY);
  gridSize.z = 1 + ( kEnd - kStart - 1 * overlapZ - 1) / (blockSize.z - 1 * overlapZ);
  return gridSize;
}

template<typename T, Processor processor, int Length>
__host__ __device__ dim3
Grid3D<T, processor, Length>::
totalCover(const dim3 blockSize, const int overlapX, const int overlapY, const int overlapZ) const {
  return cover(blockSize, 0, Nx_, 0, Ny_, 0, Nz_, overlapX, overlapY, overlapZ);
}

template<typename T, Processor processor, int Length>
__host__ __device__ dim3
Grid3D<T, processor, Length>::
excludeCover(const dim3 blockSize, const int band, const int overlapX, const int overlapY, const int overlapZ) const {
  return cover(blockSize, band, Nx_ - band, band, Ny_ - band, band, Nz_ - band, overlapX, overlapY, overlapZ);
}

template<typename T, Processor processor, int Length>
void
Grid3D<T, processor, Length>::
free() {
  switch (processor) {
    case GPU: cudaFree(this->grid); break;
    case CPU: delete[] this->grid; break;
  }
}

