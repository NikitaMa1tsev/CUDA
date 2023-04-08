#pragma once
#include "iostream"
#include "fstream" 
#include <stdlib.h>
#include <time.h>
#include <intrin.h> // cpu info
#include "vector"
#include "string"
#include "cmath"
//#include "half.hpp"
#include "chrono"
#include <iomanip> // точность

template <typename T>
auto timeMes(T*& A, int n, void (*func)(T*&, int))
{
	auto start = std::chrono::steady_clock::now();
	func(A, n);
	auto end = std::chrono::steady_clock::now();
	auto dif = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time = " << dif.count() << std::endl;
	return dif.count();
}

template <typename T>
auto timeMes(T* A, T* B, T*& C, int n, void (*func)(T*, T*, T*&, int))
{
	auto start = std::chrono::steady_clock::now();
	func(A, B, C, n);
	auto end = std::chrono::steady_clock::now();
	auto dif = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time = " << dif.count() << std::endl;
	return dif.count();
}

template <typename T>
auto timeMes(T* A, T* B, T& C, int n, void (*func)(T*, T*, T&, int))
{
	auto start = std::chrono::steady_clock::now();
	func(A, B, C, n);
	auto end = std::chrono::steady_clock::now();
	auto dif = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time = " << dif.count() << std::endl;
	return dif.count();
}


void checkSysCpu()
{
	int CPUInfo[4] = { -1 };

	__cpuid(CPUInfo, 0x80000000);

	unsigned int ID = CPUInfo[0];

	char CPUName[0x40] = { 0 };

	for (unsigned int i = 0x80000000; i <= ID; ++i)
	{
		__cpuid(CPUInfo, i);

		if (i == 0x80000002)
			memcpy(CPUName, CPUInfo, sizeof(CPUInfo));

		else if (i == 0x80000003)
			memcpy(CPUName + 16, CPUInfo, sizeof(CPUInfo));

		else if (i == 0x80000004)
			memcpy(CPUName + 32, CPUInfo, sizeof(CPUInfo));
	}

	std::cout << "CPU name: " << CPUName;
}

template <typename T>
void fillingArray(T*& A, int n)
{

	if (sizeof(T) < 4)
	{
		T a(-10), b(10);
		for (int i = 0; i < n; i++)
			A[i] = ((T)rand() / (T)RAND_MAX) * (b - a) + a;
	}

	else
	{
		for (int k = 0; k < n; k++)
		{
			T num;
			int* pNum = (int*)&num;
			int i;

			*pNum = 0;
			for (auto i = (sizeof(T) / sizeof(int)); i > 0; i--)
			{
				for (int j = sizeof(int) * 8; j > 0; j--)
				{
					*pNum <<= 1;
					*pNum |= rand() % 2;
				}
				pNum++;
			}
			if ((i = sizeof(T) % sizeof(int)) != 0)
			{
				for (i *= 8; i > 0; i--)
				{
					*pNum <<= 1;
					*pNum |= rand() % 2;
				}
			}
			A[k] = num;
		}
	}
}


template <typename T>
void outputArray(T* A, int n, std::ofstream& myfile)
{
	for (int i = 0; i < n; i++)
	{
		std::cout << A[i] << std::endl;
		myfile << "<p><span style='font-weight: bold'>" << i << "</span>-<span>" << A[i] << "</span></p>";
	}
	std::cout << std::endl;
	myfile << "<p></p>";
}


template <typename T>
void sumArrays(T* A, T* B, T*& C, int n)
{
	for (int i = 0; i < n; i++)
		C[i] = A[i] + B[i];
}


template <typename T>
void difArrays(T* A, T* B, T*& C, int n)
{
	for (int i = 0; i < n; i++)
		C[i] = A[i] - B[i];
}


template <typename T>
void multArrays(T* A, T* B, T*& C, int n)
{
	for (int i = 0; i < n; i++)
		C[i] = A[i] * B[i];
}


template <typename T>
void divArrays(T* A, T* B, T*& C, int n)
{
	for (int i = 0; i < n; i++)
		if (B[i] > 1e-16)
			C[i] = A[i] / B[i];
}


template <typename T>
void macArray(T* A, T* B, T& D, int n)
{
	D = A[0] + B[0];
	for (int i = 1; i < n; i++)
		D = D + A[i] * B[i];
	std::cout << "\nMAC = " << D << "\n" << std::endl;
}


template <typename T>
void degree(T* A, T* B, T*& C, int n)
{
	for (int i = 0; i < n; i++)
		C[i] = pow(abs(A[i]), B[i]);
}


template <typename T>
void root(T* A, T* B, T*& C, int n)
{
	for (int i = 0; i < n; i++)
		C[i] = sqrt(abs(A[i]));
}


template <typename T>
void convolutionArrays(T* A, T* B, T*& C, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			C[i + j] += A[i] * B[j];
}


template <typename T>
void ring_bufferArray(T*& A, int n)
{
	T buf;
	int K = 2;
	for (int j = 0; j < K; j++)
	{
		buf = A[0];
		for (int i = 1; i < n; i++)
		{
			A[0] = A[i];
			A[i] = buf;
			buf = A[0];
		}
	}
}


void check_act(int& act, int first, int last)
{
	bool correct_value = false;
	std::string str;
	while (!correct_value)
	{
		try
		{
			std::cin >> str;
			act = stoi(str);

			if (act >= first && act <= last)
				correct_value = true;
			else
				std::cout << "enter the test number from " << first << " to " << last << "!!!" << std::endl;
		}
		catch (...)
		{
			std::cout << "enter the test number from " << first << " to " << last << "!!!" << std::endl;
		}
	}
}


template <typename T>
void cputest(std::ofstream& myfile)
{
	int count = 5;
	int n = 10000;
	while (n < 10001)
	{
		std::cout << "---- n = " << n << " ----" << std::endl;
		myfile << "<p><span style='font-weight: bold'></span>---- n =  <span>" << n << " ----" << "</span></p>";
		T* a = new T[n], * b = new T[n], * c = new T[n], * k = new T[2 * n - 1];
		T D;

		myfile << "<p><span style='font-weight: bold'></span>fillingArray - <span>" << timeMes(b, n, &fillingArray) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>fillingArray - <span>" << timeMes(a, n, &fillingArray) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>sumArrays - <span>" << timeMes(a, b, c, n, &sumArrays) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>difArrays - <span>" << timeMes(a, b, c, n, &difArrays) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>multArrays - <span>" << timeMes(a, b, c, n, &multArrays) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>divArrays - <span>" << timeMes(a, b, c, n, &divArrays) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>macArray - <span>" << timeMes(a, b, D, n, &macArray) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>degree - <span>" << timeMes(a, b, c, n, &degree) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>root - <span>" << timeMes(a, b, c, n, &root) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>convolutionArrays - <span>" << timeMes(a, b, c, n, &convolutionArrays) << " us" << "</span></p>";
		myfile << "<p><span style='font-weight: bold'></span>ring_bufferArray - <span>" << timeMes(a, n, &ring_bufferArray) << " us" << "</span></p>";

		///std::setprecision(n)x1x1

		n += 10000;
	}
}


void mainCpu(std::ofstream& myfile)
{
	srand(time(NULL));

	checkSysCpu();
	std::cout << "\ncpu connected success\n" << std::endl;

	std::cout << "-----1-----" << std::endl;
	myfile << "<p><span style='font-weight: bold'></span>-----half-----<span></span></p>";
	//cputest<half>(myfile);
	std::cout << "-----2-----" << std::endl;
	myfile << "<p><span style='font-weight: bold'></span>-----float-----<span></span></p>";
	cputest<float>(myfile);
	std::cout << "-----3-----" << std::endl;
	myfile << "<p><span style='font-weight: bold'></span>-----double-----<span></span></p>";
	cputest<double>(myfile);
}