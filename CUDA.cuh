#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "string"
#include "iostream"
#include "fstream" 
#include "cmath"
#include "cuda_fp16.h"
#include "chrono"
#include <iomanip>

//#define _SILENCE_AMP_DEPRECATION_WARNINGS // amp gpu info
//#include <amp.h>	

//using half_float::half;

#define N 1000 // matrix size 


template <typename T>
__global__ void sumDev(T* c, const T* a, const T* b)
{
    int tin = threadIdx.x + blockIdx.x * blockDim.x;
    while (tin < N)
    {
        c[tin] = a[tin] + b[tin];
        tin += blockDim.x * gridDim.x;
    }
}


template <typename T>
__global__ void difDev(T* c, const T* a, const T* b)
{
    int tin = threadIdx.x + blockIdx.x * blockDim.x;
    while (tin < N)
    {
        c[tin] = a[tin] - b[tin];
        tin += blockDim.x * gridDim.x;
    }
}


template <typename T>
__global__ void multDev(T* c, const T* a, const T* b)
{
    int tin = threadIdx.x + blockIdx.x * blockDim.x;
    while (tin < N)
    {
        c[tin] = a[tin] * b[tin];
        tin += blockDim.x * gridDim.x;
    }
}


template <typename T>
__global__ void divDev(T* c, const T* a, const T* b)
{
    int tin = threadIdx.x + blockIdx.x * blockDim.x;
    while (tin < N)
    {
        c[tin] = a[tin] / b[tin];
        tin += blockDim.x * gridDim.x;
    }
}


static const int blockSize = 1024;

template <typename T>
__global__ void macDev(T* c, const T* a, T* b) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x * blockSize;
    const int gridSize = blockSize * gridDim.x;
    int sum = 0;
    for (int i = gthIdx; i < N; i += gridSize)
        sum += a[i] * b[i];
    __shared__ int shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx + size];
        __syncthreads();
    }
    if (thIdx == 0)
        c[blockIdx.x] = shArr[0];
}


template <typename T>
__global__ void rootDev(T* c, T* a)
{
    int tin = threadIdx.x + blockIdx.x * blockDim.x;
    while (tin < N)
    {
        c[tin] = sqrt((T)abs(a[tin]));
        tin += blockDim.x * gridDim.x;
    }
}


template <typename T>
__global__ void degreeDev(T* c, T* a, T* b)
{
    int tin = threadIdx.x + blockIdx.x * blockDim.x;
    while (tin < N)
    {
        c[tin] = powf(a[tin], b[tin]);
        tin += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void convolutionDev(T* c, T* a, T* b) {

    int tin = blockIdx.x * blockDim.x + threadIdx.x;

    T temp = 0;

    int n_start_point = tin - (N / 2);

    for (int j = 0; j < N; j++) {
        if (n_start_point + j >= 0 && n_start_point + j < N) {
            temp += a[n_start_point + j] * b[j];
        }
    }

    c[tin] = temp;
}



//template <typename T>
//__global__ void ring_bufferDev(T*c, T* a, int K)
//{
//    int tin = threadIdx.x + blockIdx.x * blockDim.x;
//
//    T buf;
//  
//    for (int j = 0; j < K; j++)
//    {
//        //if (threadIdx.x == 0)
//        //    a[tin] = a[tin + N -1];
//        if (tin < N)
//        {
//            a[tin+1] = a[tin];
//            tin += blockDim.x * gridDim.x;
//        }
//        if (threadIdx.x == 0)
//            a[tin] = a[tin + N -1];
//        __syncthreads();
//    }
//}