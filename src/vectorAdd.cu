// vectorAdd.cu - Suma de vectores C = A + B en CPU y en GPU (CUDA)
// Uso: ./vectorAdd [N]        (N por defecto = 16M = 1 << 24)
// Reporta: tiempo CPU, tiempo del kernel y tiempo GPU total (H2D + kernel + D2H)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Verifica el resultado de cada llamada a la API de CUDA
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err_));                  \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Verifica errores de lanzamiento de un kernel
#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

// Kernel CUDA: cada thread calcula un elemento de C
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Suma de vectores en CPU (referencia)
void vectorAddCPU(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

// Tiempo de reloj de pared en milisegundos
static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    if (n <= 0) { fprintf(stderr, "N invalido\n"); return 1; }
    size_t bytes = (size_t)n * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | N = %d elementos (%.1f MB por vector)\n",
           prop.name, n, bytes / (1024.0 * 1024.0));

    // Memoria en host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) { fprintf(stderr, "malloc fallo\n"); return 1; }

    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
    }

    // --- Benchmark CPU ---
    double t0 = wall_ms();
    vectorAddCPU(h_A, h_B, h_C_cpu, n);
    double cpu_ms = wall_ms() - t0;

    // --- Benchmark GPU ---
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    // Calentamiento: el primer lanzamiento incluye la inicializacion del contexto
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evIni, evH2D, evKer, evFin;
    CUDA_CHECK(cudaEventCreate(&evIni));
    CUDA_CHECK(cudaEventCreate(&evH2D));
    CUDA_CHECK(cudaEventCreate(&evKer));
    CUDA_CHECK(cudaEventCreate(&evFin));

    CUDA_CHECK(cudaEventRecord(evIni));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(evH2D));
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(evKer));
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(evFin));
    CUDA_CHECK(cudaEventSynchronize(evFin));

    float h2d_ms, ker_ms, d2h_ms, tot_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, evIni, evH2D));
    CUDA_CHECK(cudaEventElapsedTime(&ker_ms, evH2D, evKer));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, evKer, evFin));
    CUDA_CHECK(cudaEventElapsedTime(&tot_ms, evIni, evFin));

    printf("CPU:                %8.2f ms\n", cpu_ms);
    printf("GPU kernel:         %8.3f ms  (%.1f GB/s efectivos)\n",
           ker_ms, 3.0 * bytes / (ker_ms * 1e6));
    printf("GPU copia H2D:      %8.2f ms\n", h2d_ms);
    printf("GPU copia D2H:      %8.2f ms\n", d2h_ms);
    printf("GPU total (con memcpy): %4.2f ms\n", tot_ms);
    printf("Speedup kernel: %.1fx | Speedup total: %.2fx\n",
           cpu_ms / ker_ms, cpu_ms / tot_ms);

    // Verificar resultados
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (fabsf(h_C_gpu[i] - h_C_cpu[i]) > 1e-4f) errors++;
    printf("Errores: %d\n", errors);

    // Liberar recursos
    CUDA_CHECK(cudaEventDestroy(evIni)); CUDA_CHECK(cudaEventDestroy(evH2D));
    CUDA_CHECK(cudaEventDestroy(evKer)); CUDA_CHECK(cudaEventDestroy(evFin));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return errors == 0 ? 0 : 1;
}
