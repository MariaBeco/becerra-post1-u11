// matMul.cu - Multiplicacion de matrices C = A x B: naive vs tiling con shared memory
// Uso: ./matMul [N]   (sin argumento ejecuta N = 512 y N = 1024)
// Cada version se compara contra una multiplicacion de referencia en CPU.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE 16
#define REPS 10   // repeticiones de cada kernel para promediar

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__,        \
                    __LINE__, cudaGetErrorString(err_));                  \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

// Version naive: cada thread lee una fila de A y una columna de B
// directamente de memoria global (2N lecturas globales por elemento de C)
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// Version con tiling: el bloque carga tiles TILE x TILE en shared memory y
// cada valor leido de memoria global se reutiliza TILE veces
__global__ void matMulTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // Cargar tile de A en shared memory
        if (row < N && t * TILE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        // Cargar tile de B en shared memory
        if (col < N && t * TILE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();   // Esperar a que el tile este cargado
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();   // Esperar antes de sobrescribir el tile
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

// Referencia en CPU (orden i-k-j para recorrer B por filas)
void matMulCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float a = A[i * N + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
}

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float maxAbsError(const float *x, const float *ref, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(x[i] - ref[i]);
        if (e > m) m = e;
    }
    return m;
}

// Ejecuta un kernel REPS veces y devuelve el tiempo promedio en ms
typedef void (*kernel_t)(const float*, const float*, float*, int);
static float timeKernel(kernel_t k, dim3 grid, dim3 block,
                        const float *d_A, const float *d_B, float *d_C, int N) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    k<<<grid, block>>>(d_A, d_B, d_C, N);          // calentamiento
    KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < REPS; r++) k<<<grid, block>>>(d_A, d_B, d_C, N);
    KERNEL_CHECK();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / REPS;
}

static int runSize(int N) {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes), *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_ref || !h_C) { fprintf(stderr, "malloc fallo\n"); exit(1); }

    // Valores multiplos de 1/4 y 1/2: todas las sumas parciales son exactas
    // en float, asi CPU y GPU deben coincidir aunque sumen en distinto orden
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(i % 7) * 0.25f;
        h_B[i] = (float)(i % 5) * 0.5f;
    }

    double t0 = wall_ms();
    matMulCPU(h_A, h_B, h_ref, N);
    double cpu_ms = wall_ms() - t0;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Lanzamiento: grid 2D de tiles
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    double gflop = 2.0 * N * N * (double)N / 1e9;

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    float naive_ms = timeKernel(matMulNaive, grid, block, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    float errNaive = maxAbsError(h_C, h_ref, N * N);

    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    float tiled_ms = timeKernel(matMulTiled, grid, block, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    float errTiled = maxAbsError(h_C, h_ref, N * N);

    printf("\nN = %d  (%.2f GFLOP por multiplicacion)\n", N, gflop);
    printf("  CPU referencia: %10.2f ms  (%7.2f GFLOPS)\n", cpu_ms, gflop / (cpu_ms / 1e3));
    printf("  GPU naive:      %10.3f ms  (%7.1f GFLOPS)  error max = %g\n",
           naive_ms, gflop / (naive_ms / 1e3), errNaive);
    printf("  GPU tiled:      %10.3f ms  (%7.1f GFLOPS)  error max = %g\n",
           tiled_ms, gflop / (tiled_ms / 1e3), errTiled);
    printf("  Speedup tiled vs naive: %.2fx | tiled vs CPU: %.0fx\n",
           naive_ms / tiled_ms, cpu_ms / tiled_ms);
    int ok = errNaive < 1e-3f && errTiled < 1e-3f;
    printf("  Verificacion (error < 1e-3): %s\n", ok ? "OK" : "FALLA");

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_ref); free(h_C);
    return ok;
}

int main(int argc, char **argv) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | TILE = %d | promedio de %d ejecuciones por kernel\n",
           prop.name, TILE, REPS);
    int ok = 1;
    if (argc > 1) {
        ok = runSize(atoi(argv[1]));
    } else {
        ok &= runSize(512);
        ok &= runSize(1024);
    }
    return ok ? 0 : 1;
}
