#include "CUDA.cuh"
#include "CPU.h"

//#define BLOCK_SIZE 16 // submatrix size
#define N 1000000 // matrix size 



template <typename T>
void GPUmain()
{
    int numBytes =  N*sizeof(T);

    T* a = new T[N], *b = new T[N], *c = new T[N];
    T Mac;

    fillingArray(a, N);
    fillingArray(b, N);
    fillingArray(c, N);

   // for (int i = 0; i < N; i++)
   //     std::cout << std::setprecision(12) << a[i] << std::endl;
   // printf("\n");

    T* dev_a = NULL;
    T* dev_b = NULL;
    T* dev_c = NULL;
 

    cudaMalloc((void**)&dev_c, numBytes);
    cudaMalloc((void**)&dev_b, numBytes);
    cudaMalloc((void**)&dev_a, numBytes);
   // cudaMalloc((void**)&dev_mac, numBytes);

    // set kernel launch configuration

    //dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 blocks(N / threads.x, N / threads.y);

    // create cuda event handles

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // asynchronously issue work to the GPU (all to stream 0)

    cudaEventRecord(start, 0);
    cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, numBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    
    //----macDev<<<blocksPerGrid, threadsPerBlock>>> (dev_c, dev_a, dev_b);
    
    //macDev <<<1, threadsPerBlock >>> (dev_c, dev_a, dev_b);
    //cudaMemcpy(c, dev_c, sizeof(T), cudaMemcpyDeviceToHost);
    // 
    //rootDev<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a);

    //degreeDev<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b);

    //convolutionDev<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b);
    

    //ring_bufferDev<<<1, threadsPerBlock >> > (dev_c, dev_a, 1);
    //cudaMemcpy(a, dev_a, N * sizeof(T), cudaMemcpyDeviceToHost);

    //cudaMemcpy(c, dev_c, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

   
    //--------
    // print the cpu and gpu times

    printf("time spent executing by the GPU: %.2f millseconds\n", gpuTime);
    //for (int i= 0; i < N; i++)
    //  std::cout << std::setprecision(12) << a[i] << std::endl;


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);


    delete[] a;
    delete[] b;
    delete[] c;
}

void print_cuda_device_info(cudaDeviceProp &prop)
{
    printf("%i\n", prop.warpSize);
}



int main()
{
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);
    printf("%s\n", prop.name, 0);
    printf("%i\n", prop.maxThreadsPerBlock, 0);
    printf("%i\n", prop.regsPerBlock, 0);

 

    int act; bool loop = true;
    std::ofstream myfile;
    myfile.open("D:\\Program\\Benchmark\\report.html");
    myfile << "<!DOCTYPE html><html><head><h1>Результаты работы</h1></head><body>";

    while (loop)
    {
        system("cls");
        std::cout << "Menu:\n1. Test_CPU\n2. Test_GPU\n3. Exit\n";

        check_act(act, 1, 3);

        switch (act)
        {
        case 1:
        {
            std::cout << "selected CPU" << std::endl;
            mainCpu(myfile);
        }; break;

        case 2:
        {
            std::cout << "selected GPU" << std::endl;
            //GPUmain<half>();
            std::cout << "float" << std::endl;
            GPUmain<float>();
            std::cout << "double" << std::endl;
            GPUmain<double>();
        }; break;

        case 3: loop = false; break;
        }
        system("pause");
    }

    myfile << "</body></html>";
    myfile.close();
}
