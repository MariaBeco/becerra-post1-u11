# Post-Contenido 1: CUDA Benchmark CPU vs GPU

Arquitectura de Computadores, Unidad 11
Maria Fernanda Becerra Coronel, código 1152509, Universidad Francisco de Paula Santander
Docente: Ing. Jonathan Rey

Este repositorio contiene las dos partes del laboratorio. La Parte 1 implementa y mide tres programas CUDA en C: suma de vectores, multiplicación de matrices con tiling en shared memory y suma de vectores con streams. La Parte 2 es el análisis comparativo de aceleradores de IA, en [`docs/analisis-comparativo.md`](docs/analisis-comparativo.md).

## Estructura

```
becerra-post1-u11/
├── README.md
├── src/
│   ├── vectorAdd.cu          suma de vectores, CPU vs GPU, N configurable
│   ├── matMul.cu             multiplicación de matrices naive vs tiling (N=512 y N=1024)
│   └── vectorAddStream.cu    suma de vectores síncrona vs con streams
├── docs/
│   └── analisis-comparativo.md
└── capturas/
```

## Entorno

El equipo de la estudiante no tiene GPU NVIDIA, por lo que los programas se ejecutaron en Google Colab, como permite la guía.

| Elemento | Valor |
|---|---|
| Plataforma | Google Colab, entorno con GPU |
| GPU | NVIDIA Tesla T4 (Turing, 15 360 MiB de memoria GDDR6, 70 W) |
| Driver / CUDA (nvidia-smi) | Driver 580.82.07, CUDA 13.0 |
| Compilador | nvcc 13.0, V13.0.88 |
| Sistema operativo | Linux (máquina virtual de Google Colab) |
| Copy engines (asyncEngineCount) | 3 |

## Compilación y ejecución

```bash
nvcc -O2 -arch=sm_75 -o vectorAdd src/vectorAdd.cu
nvcc -O2 -arch=sm_75 -o matMul src/matMul.cu
nvcc -O2 -arch=sm_75 -o vectorAddStream src/vectorAddStream.cu

./vectorAdd 1048576        # N = 1M
./vectorAdd 4194304        # N = 4M
./vectorAdd 16777216       # N = 16M (valor por defecto)
./matMul                   # N = 512 y N = 1024
./vectorAddStream          # 4 streams (valor por defecto)
./vectorAddStream 8        # otro número de streams
```

`-arch=sm_75` genera código para la arquitectura Turing de la T4. Sin esa opción, las versiones recientes de nvcc compilan para una arquitectura más antigua y muestran una advertencia de que su soporte será retirado.

## Cambios respecto al código de la guía

El código de la guía no compila tal como está y mide algunos tiempos de forma incompleta. Se hicieron estos cambios:

1. **Includes faltantes.** `vectorAdd.cu` usa `fabs` y `clock` sin incluir `<math.h>` ni `<time.h>`. Se agregaron, y se usa `fabsf`, la versión de `float`.
2. **Manejo de errores.** Todas las llamadas a la API pasan por la macro `CUDA_CHECK`, y después de cada lanzamiento de kernel se llama a `cudaGetLastError()`. Sin esto, un fallo de la GPU pasa inadvertido y el programa imprime tiempos de un kernel que nunca se ejecutó.
3. **Tiempo total de GPU.** La guía mide solo el kernel, pero el README pide también el tiempo total con `cudaMemcpy`. `vectorAdd.cu` registra cuatro eventos y reporta por separado la copia H2D, el kernel, la copia D2H y el total.
4. **Calentamiento.** El primer lanzamiento de un kernel incluye la inicialización del contexto CUDA, que tarda decenas o cientos de milisegundos. Los tres programas lanzan el kernel una vez antes de medir.
5. **Tiempo de CPU.** `clock()` mide tiempo de procesador, no tiempo real. Se reemplazó por `clock_gettime(CLOCK_MONOTONIC)`.
6. **Tamaño por argumento.** `vectorAdd` recibe N por la línea de comandos, para medir 1M, 4M y 16M sin recompilar. `vectorAddStream` recibe el número de streams, para la Decisión 2.
7. **matMul completo.** La guía solo trae el kernel con tiling. Se agregaron el kernel naive, la referencia en CPU, la verificación del error y la medición de ambos kernels con promedio de 10 ejecuciones. Las matrices usan valores múltiplos de 1/4 y de 1/2, con los que todas las sumas son exactas en `float`; así la CPU y la GPU deben coincidir aunque sumen en distinto orden.
8. **Verificación en streams.** La guía no comprueba el resultado de `vectorAddStream.cu`. El programa limpia `h_C` antes de cada versión y compara ambas contra la suma calculada en CPU. También admite un N que no sea múltiplo del número de streams, y promedia 5 ejecuciones de cada versión.

## Resultados

### vectorAdd

| N | CPU (ms) | GPU kernel (ms) | Copia H2D (ms) | Copia D2H (ms) | GPU total (ms) | Speedup kernel | Speedup total |
|---|---|---|---|---|---|---|---|
| 1M (1 048 576) | 2.58 | 0.054 | 2.66 | 2.91 | 5.63 | 48.3× | 0.46× |
| 4M (4 194 304) | 9.38 | 0.191 | 7.98 | 11.86 | 20.04 | 49.1× | 0.47× |
| 16M (16 777 216) | 40.68 | 0.767 | 29.85 | 45.50 | 76.11 | 53.0× | 0.53× |

### matMul (promedio de 10 ejecuciones por kernel)

| N | CPU (ms) | GPU naive (ms) | GPU tiled (ms) | Speedup tiled vs naive | GFLOPS tiled | Error máximo |
|---|---|---|---|---|---|---|
| 512 | 109.68 | 1.147 | 0.677 | 1.69× | 396.2 | 0 |
| 1024 | 802.99 | 9.183 | 5.801 | 1.58× | 370.2 | 0 |

### vectorAddStream (N = 16M, promedio de 5 ejecuciones)

| NSTREAMS | Síncrono (ms) | Con streams (ms) | Reducción |
|---|---|---|---|
| 1 | 16.86 | 16.88 | −0.1 % |
| 2 | 16.89 | 14.61 | 13.5 % |
| 4 | 16.89 | 13.57 | 19.7 % |
| 8 | 16.89 | 13.29 | 21.3 % |
| 16 | 16.88 | 13.23 | 21.6 % |

En la ejecución individual de `./vectorAddStream` (captura 5), con 4 streams, el resultado fue 16.88 ms síncrono, 13.51 ms con streams y una reducción del 20.0 %. Con 1 stream ambas versiones hacen lo mismo y tardan igual, lo que confirma que la medición no favorece a ninguna de las dos.

## Análisis

**El kernel es mucho más rápido que la CPU.** El kernel de vectorAdd es entre 48 y 53 veces más rápido que el bucle de la CPU. La CPU suma los elementos con uno o pocos núcleos, mientras que la T4 lanza un thread por elemento y mantiene miles de ellos activos en sus 40 SM, lo que le permite ocultar la latencia de la memoria cambiando de warp. Como vectorAdd hace una sola suma por cada 12 bytes que mueve, su límite es el ancho de banda de la memoria, no el cómputo. Con N = 16M, el kernel movió 192 MiB en 0.767 ms, unos 262 GB/s efectivos, el 82 % de los 320 GB/s de la GDDR6 de la T4. La ventaja crece un poco con N (de 48× a 53×) porque con vectores grandes el costo fijo de lanzar el kernel pesa menos.

**El tiempo total con copias es mayor que el de la CPU.** Al incluir `cudaMemcpy`, la GPU tarda el doble que la CPU en los tres tamaños (speedup total de 0.46× a 0.53×). Con N = 16M, el kernel ocupa solo 0.77 ms de los 76.11 ms totales, es decir, el 1 %; el resto son copias por PCIe. Esas copias son lentas porque `vectorAdd.cu` reserva la memoria del host con `malloc`, que es memoria paginable: el driver primero copia los datos a un buffer fijado y desde ahí los envía por DMA. Así, la copia H2D llegó a 4.5 GB/s y la D2H a solo 1.5 GB/s. En `vectorAddStream.cu`, que usa memoria fijada con `cudaHostAlloc`, los mismos 192 MiB de ida y vuelta tardan 16.1 ms (unos 12.5 GB/s); con ese tipo de memoria, el total de 16.88 ms ya sería 2.4 veces más rápido que la CPU. Aun así, para una operación con tan poco cómputo por byte, la GPU solo conviene si los datos ya están en su memoria o si se reutilizan en varios kernels; si hay que copiarlos para hacer una sola suma, la CPU termina antes. Con N pequeño la situación es peor, porque las copias tienen además un costo fijo de inicio que no depende del tamaño.

**El tiling mejora la multiplicación de matrices, pero menos de lo que predice la teoría.** Con TILE = 16, el kernel tiled fue 1.69 veces más rápido que el naive con N = 512 y 1.58 veces con N = 1024, y ambos coincidieron exactamente con la CPU (error máximo 0). En teoría, el tiling reduce 16 veces las lecturas a memoria global, pero la mejora medida es mucho menor porque el kernel naive no lee todo de la DRAM: los threads de un warp leen las mismas filas de A y columnas consecutivas de B, y las cachés L1 y L2 de la T4 sirven gran parte de esas lecturas repetidas. El kernel tiled llega a 370–396 GFLOPS, alrededor del 5 % de los 8.1 TFLOPS FP32 de la T4, porque cada thread hace dos lecturas de shared memory por cada multiplicación-suma y espera en dos `__syncthreads()` por tile. Aun así, frente a la CPU, que solo alcanzó unos 2.5 GFLOPS con un bucle secuencial, la GPU fue entre 138 y 162 veces más rápida.

## Decisión 1: ¿cudaMemcpyAsync con streams o transferencia síncrona para vectorAdd?

**Decisión: para vectorAdd, los streams solo dan una mejora modesta, y la mayor parte no viene de ocultar el cómputo. Donde más rinden es en operaciones como matMulTiled.**

El volumen transferido con N = 16M es de 192 MiB: dos vectores de entrada de 64 MiB hacia la GPU y uno de 64 MiB de vuelta. A 10–12 GB/s efectivos, eso toma entre 17 y 20 ms. En la T4 de Colab, que usa PCIe 3.0 x16, las copias con memoria fijada tomaron 16.1 ms (16.88 ms totales menos 0.77 ms de kernel), unos 12.5 GB/s. El kernel tarda 0.767 ms: las transferencias duran 21 veces más que el cómputo, así que vectorAdd es claramente memory-bound.

Si los streams solo solaparan copia con cómputo, el ahorro máximo sería el tiempo del kernel, porque el solapamiento solo puede esconder la fase más corta detrás de la más larga: 0.77 ms de 16.88 ms, un 4.5 %. Sin embargo, la reducción medida fue del 20 %. La diferencia se explica porque PCIe es full-duplex y la T4 tiene motores de copia separados: con varios chunks, la copia D2H del chunk 1 avanza al mismo tiempo que la copia H2D del chunk 2. Lo que se solapa no es tanto el cómputo como las dos direcciones de transferencia. Aun así, el límite sigue siendo la fase más larga: la copia H2D de 128 MiB, que debería tomar unos 10.7 ms si el bus rinde igual en ambas direcciones, no se puede ocultar detrás de nada, así que el tiempo total no puede bajar de ese valor.

En matMulTiled la situación es la opuesta. Con N = 1024, cada matriz ocupa 4 MiB, así que las tres copias suman 12 MiB, alrededor de 1 ms a 12.5 GB/s, mientras que el kernel tarda 5.8 ms, porque hace O(N³) operaciones sobre O(N²) datos. Ahí el cómputo domina, y dividir el trabajo en chunks permite esconder casi toda la transferencia detrás del cálculo. Por eso se mantienen los streams en vectorAdd, porque el 20 % es real y el código es sencillo, pero se reconoce que la técnica rinde mucho más en operaciones con alta intensidad aritmética.

## Decisión 2: ¿cuántos streams usar, y qué limita la ganancia al aumentar su número?

**Decisión: usar 4 streams. Más allá de ese número la ganancia es marginal, porque el límite no lo pone la cantidad de streams sino los motores de copia y el ancho de banda de PCIe.**

Los resultados no mejoran de forma proporcional al número de streams: con 2 streams la reducción es del 13.5 %, con 4 del 19.7 %, con 8 del 21.3 % y con 16 del 21.6 %. Pasar de 4 a 16 streams cuadruplica la división del trabajo y solo agrega 1.9 puntos. El tiempo se estanca cerca de 13.2 ms, que está por encima del piso teórico de la copia H2D (unos 10.7 ms); la diferencia es lógica, porque las dos direcciones no conservan todo su ancho de banda cuando se transfieren al mismo tiempo.

El recurso que impone el límite es físico. La T4 reporta 3 copy engines (`asyncEngineCount = 3`), pero en la práctica el pipeline solo necesita uno en cada dirección: uno para la copia H2D de un chunk y otro para la D2H de otro chunk, mientras el motor de cómputo procesa un tercero. Con 3 o 4 streams ya hay suficiente trabajo en vuelo para mantener ocupados esos tres recursos. Un stream adicional no crea un nuevo motor DMA ni un carril más en el bus; su copia espera en la cola de un motor que ya está ocupado.

Además, dividir en más streams no reduce el volumen total: los 192 MiB tienen que cruzar el mismo bus PCIe, que es un recurso compartido por todos los streams. Más streams solo cambian cuándo se mueve cada porción de datos, lo que mejora el solapamiento al inicio y al final del pipeline, porque el primer y el último chunk son más pequeños. A cambio, cada chunk adicional agrega el costo de lanzar un kernel y tres copias más. De 8 a 16 streams, ese costo casi anula la mejora adicional. Por eso 4 streams es el punto de equilibrio en esta GPU: obtiene el 91 % de la mejora máxima medida con la cuarta parte de las operaciones de 16 streams.

## Capturas

| Captura | Contenido |
|---|---|
| `capturas/cap01_entorno.png` | `nvidia-smi` y `nvcc --version` |
| `capturas/cap02_compilacion.png` | Compilación de los tres programas sin advertencias |
| `capturas/cap03_vectorAdd.png` | Checkpoint 1: vectorAdd con N = 1M, 4M y 16M |
| `capturas/cap04_matMul.png` | Checkpoint 2: matMul naive vs tiled, N = 512 y 1024 |
| `capturas/cap05_streams.png` | Checkpoint 4: síncrono vs 4 streams |
| `capturas/cap06_streams_variando.png` | Decisión 2: 1, 2, 4, 8 y 16 streams |
| `capturas/cap07a_readme_resultados.png` | Checkpoint 3: tablas de resultados del README (vectorAdd y matMul) |
| `capturas/cap07b_readme_resultados.png` | Checkpoint 3: tablas de resultados del README (vectorAddStream) |
| `capturas/cap08_decision1.png` | Checkpoint 5: Decisión 1 |
| `capturas/cap09_decision2.png` | Checkpoint 5: Decisión 2 |
