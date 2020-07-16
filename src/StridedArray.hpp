#pragma once

template<typename T, int Length>
class StridedArray {
protected:
	const int stride;
	T* const cell;
	int* const counter;
	
public:
	__host__  __forceinline__ StridedArray() :
		cell(new T[Length]),
		stride(1),
		counter(new int) {
		*counter = 1;
	}

	__host__ __device__ ~StridedArray() {
		if (counter && --*counter == 0) {
			delete counter;
			delete[] cell;
		}
	}

	__host__ __device__ __forceinline__ StridedArray(const StridedArray<T, Length>& copy) :
		cell(copy.cell),
		counter(copy.counter),
		stride(copy.stride) {
		if (counter) ++*counter;
	}

	__host__ __device__ __forceinline__ StridedArray(T* const cell, const int stride = 1) :
		cell(cell),
		stride(stride),
		counter(NULL) {
	}

	__host__ __device__ __forceinline__ T& operator[](const int v) {
		return cell[v * stride];
	}

	__host__ __device__ __forceinline__ T operator[](const int v) const {
		return cell[v * stride];
	}

	__host__ __device__  __forceinline__ StridedArray<T, Length>& operator=(const StridedArray<T, Length>& copy) {
		for (int k = 0; k < Length; k++) {
			(*this)[k] = copy[k];
		}
		return *this;
	}

	__host__ __device__ __forceinline__ StridedArray<T, Length>& operator=(const float& a) {
		for (int k = 0; k < Length; k++) {
			(*this)[k] = a;
		}
		return *this;
	}

	__host__ __device__ __forceinline__ StridedArray<T, Length>& operator+=(const StridedArray<T, Length>& a) {
		for (int k = 0; k < Length; k++) {
			(*this)[k] += a[k];
		}
		return *this;
	}

	__host__ __device__ __forceinline__ static int length() { return Length; }
};

template<typename T, int Length>
class StridedCell: public StridedArray<T, Length> {
public:
	typedef StridedArray<T, Length> Parent;

	__host__  __forceinline__ StridedCell() : Parent() { }

	__host__ __device__ ~StridedCell() { }

	using Parent::operator=;
	__host__ __device__ StridedCell<T, Length>& operator=(const StridedCell<T, Length>& copy) {
		Parent::operator=(copy);
		return *this;
	}

	__host__ __device__ __forceinline__ StridedCell(const StridedCell<T, Length>& copy) : Parent(copy) { }

	__host__ __device__ __forceinline__ StridedCell(T* const cell, const int stride = 1) : Parent(cell, stride) { }

	__host__ __device__ T& temperature() {
		return (*this)[TG];
	}
	__host__ __device__ T temperature() const {
		return (*this)[TG];
	}
	__host__ __device__ T& concentration() {
		return (*this)[CG];
	}
	__host__ __device__ T concentration() const {
		return (*this)[CG];
	}
};

template<typename T, int Length, typename S>
StridedArray<T, Length> operator*(const S& a, const StridedArray<T, Length>& b) {
	StridedArray<T, Length> c;
	for (int k = 0; k < b.length; k++) {
		c[k] = a * b[k];
	}
	return c;
}

template<typename T, int Length>
StridedArray<T, Length> operator+(const StridedArray<T, Length>& a, const StridedArray<T, Length>& b) {
	StridedArray<T, Length> c;
	for (int k = 0; k < b.length; k++) {
		c[k] = a[k] + b[k];
	}
	return c;
}

template<typename T, int Length>
StridedArray<T, Length> operator-(const StridedArray<T, Length>& a, const StridedArray<T, Length>& b) {
	StridedArray<T, Length> c;
	for (int k = 0; k < b.length; k++) {
		c[k] = a[k] - b[k];
	}
	return c;
}

template<typename T, int Length, typename S>
StridedArray<T, Length> operator/(const StridedArray<T, Length>& a, const S& b) {
	StridedArray<T, Length> c;
	for (int k = 0; k < a.length; k++) {
		c[k] = a[k] / b;
	}
	return c;
}
