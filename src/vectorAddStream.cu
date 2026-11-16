// vectorAddStream.cu - vectorAdd sincrono vs con streams (copia y computo solapados)
// Uso: ./vectorAddStream [NSTREAMS]   (por defecto 4)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define N (1 << 24)        // 16M elementos, igual que vectorAdd.cu
#define REPS 5             // repeticiones de cada version; se reporta el promedio

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

__global__ void vectorAddChunk(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// Compara contra la suma calculada en CPU
static int countErrors(const float *h_C, const float *h_ref) {
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_C[i] - h_ref[i]) > 1e-4f) errors++;
    return errors;
}

int main(int argc, char **argv) {
    int nStreams = (argc > 1) ? atoi(argv[1]) : 4;
    if (nStreams < 1 || nStreams > 64) { fprintf(stderr, "NSTREAMS debe estar entre 1 y 64\n"); return 1; }
    size_t bytes = (size_t)N * sizeof(float);
    int chunk = (N + nStreams - 1) / nStreams;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | copy engines (asyncEngineCount): %d | NSTREAMS = %d\n",
           prop.name, prop.asyncEngineCount, nStreams);

    // Memoria host "pinned" (page-locked): obligatoria para que
    // cudaMemcpyAsync se ejecute realmente de forma asincrona
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaHostAlloc(&h_A, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_B, bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_C, bytes, cudaHostAllocDefault));
    float *h_ref = (float*)malloc(bytes);
    if (!h_ref) { fprintf(stderr, "malloc fallo\n"); return 1; }
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i * 0.5f;
        h_B[i] = (float)i * 1.5f;
        h_ref[i] = h_A[i] + h_B[i];   // referencia en CPU
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int blockSize = 256;

    cudaStream_t *streams = (cudaStream_t*)malloc(nStreams * sizeof(cudaStream_t));
    for (int s = 0; s < nStreams; s++) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // Calentamiento (inicializa el contexto y carga el kernel)
    vectorAddChunk<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_A, d_B, d_C, N);
    KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Baseline sincrono: una sola transferencia H2D, un kernel, una D2H ---
    float syncMs = 0.0f;
    for (int r = 0; r < REPS; r++) {
        memset(h_C, 0, bytes);
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        vectorAddChunk<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_A, d_B, d_C, N);
        KERNEL_CHECK();
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        syncMs += ms;
    }
    syncMs /= REPS;
    int errSync = countErrors(h_C, h_ref);
    printf("Sincrono (H2D + kernel + D2H completos): %.2f ms | errores: %d\n", syncMs, errSync);

    // --- Version con streams: solapa transferencia y computo por chunks ---
    float asyncMs = 0.0f;
    for (int r = 0; r < REPS; r++) {
        memset(h_C, 0, bytes);
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        CUDA_CHECK(cudaEventRecord(start));
        for (int s = 0; s < nStreams; s++) {
            size_t off = (size_t)s * chunk;
            int len = (off + chunk <= (size_t)N) ? chunk : (int)(N - off);
            if (len <= 0) break;
            size_t lenBytes = (size_t)len * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(d_A + off, h_A + off, lenBytes,
                                       cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(d_B + off, h_B + off, lenBytes,
                                       cudaMemcpyHostToDevice, streams[s]));
            vectorAddChunk<<<(len + blockSize - 1) / blockSize, blockSize, 0, streams[s]>>>(
                d_A + off, d_B + off, d_C + off, len);
            KERNEL_CHECK();
            CUDA_CHECK(cudaMemcpyAsync(h_C + off, d_C + off, lenBytes,
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        for (int s = 0; s < nStreams; s++) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        asyncMs += ms;
    }
    asyncMs /= REPS;
    int errAsync = countErrors(h_C, h_ref);
    printf("Con streams (%d, solapado): %.2f ms | errores: %d\n", nStreams, asyncMs, errAsync);
    printf("Reduccion: %.1f%%  (promedio de %d ejecuciones)\n",
           100.0 * (syncMs - asyncMs) / syncMs, REPS);

    for (int s = 0; s < nStreams; s++) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    free(streams);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A)); CUDA_CHECK(cudaFreeHost(h_B)); CUDA_CHECK(cudaFreeHost(h_C));
    free(h_ref);
    return (errSync == 0 && errAsync == 0) ? 0 : 1;
}
