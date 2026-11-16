# Análisis comparativo de aceleradores de hardware para inteligencia artificial

**Arquitectura de Computadores, Unidad 11, Post-contenido (Parte 2)**
Maria Fernanda Becerra Coronel, código 1152509, Universidad Francisco de Paula Santander

## Introducción

Las cargas de trabajo de inteligencia artificial se reducen, en su mayor parte, a multiplicaciones de matrices y convoluciones sobre grandes volúmenes de datos. Cuatro familias de hardware compiten por ejecutarlas: la GPU, que reparte el trabajo entre miles de threads SIMT; la TPU, que lo resuelve con arreglos sistólicos; la NPU, que lo ejecuta dentro de un SoC de bajo consumo; y la FPGA, que lo implementa como un circuito configurable. Este documento compara las cuatro con datos reales. Las especificaciones provienen de las hojas técnicas de cada fabricante, y las cifras de rendimiento provienen de los resultados oficiales de MLPerf Inference v4.1 (MLCommons, 2024), tomados directamente de los registros `mlperf_log_summary.txt` publicados en el repositorio oficial de resultados. Con esos datos se analizan el costo total de propiedad, la recomendación para tres perfiles de organización y el efecto de la cuantización INT8. Al final se presentan dos decisiones técnicas sobre el Perfil B.

## 1. Caracterización de cada acelerador

| Acelerador | Paradigma | Unidad de cómputo | Memoria | Pico de cómputo | Potencia | Programación | Fuente |
|---|---|---|---|---|---|---|---|
| GPU NVIDIA H100 SXM | SIMT masivo | 132 SM con CUDA Cores y Tensor Cores de 4.ª generación | 80 GB HBM3, 3,35 TB/s; 50 MB de L2 | 989 TFLOPS FP16/BF16 y 1979 TOPS INT8 (densos) | Hasta 700 W | CUDA, cuDNN, TensorRT | NVIDIA (2022, 2023) |
| TPU v6e (Trillium) | Arreglo sistólico | 1 TensorCore con 2 MXU de 256×256 | 32 GB HBM, 1638 GB/s; ICI de 800 GB/s | 918 TFLOPS BF16 y 1836 TOPS INT8 | No publicada por chip | XLA, JAX, TensorFlow, PyTorch/XLA | Google Cloud (2025a) |
| NPU Apple M3 / M3 Ultra | Dataflow fijo en SoC | Neural Engine de 16 núcleos (M3) y de 32 núcleos (M3 Ultra) | Memoria unificada LPDDR compartida con CPU y GPU (hasta 512 GB en M3 Ultra) | Hasta 60 % más rápido que el de M1 (Apple no lo publica en TOPS) | Pocos vatios dentro del SoC | Core ML, BNNS | Apple (2023, 2025) |
| FPGA AMD Alveo U55C | Lógica reconfigurable | LUT, slices DSP y BRAM/URAM organizados por el diseñador | 16 GB HBM2, 460 GB/s | Depende del circuito sintetizado | 150 W máximo | Vitis HLS, VHDL, Verilog | AMD (2023) |

**GPU.** La H100 agrupa sus recursos en multiprocesadores (SM) que ejecutan warps de 32 threads. Cada SM incluye Tensor Cores que calculan una multiplicación de matrices pequeñas por instrucción y admiten FP64, TF32, BF16, FP16, FP8 e INT8. Su fortaleza es la generalidad: cualquier operación expresable como kernel CUDA se ejecuta, y la memoria HBM3 de 3,35 TB/s alimenta a los SM con un ancho de banda que ninguna otra familia de esta tabla iguala en un solo dispositivo (NVIDIA, 2022).

**TPU.** La TPU dedica casi todo el silicio a las MXU. En un arreglo sistólico, cada dato entra por un borde y avanza de celda en celda; cada celda multiplica y acumula sin volver a leer memoria, así que un valor se reutiliza cientos de veces. Trillium duplicó el tamaño de la MXU de 128×128 (v5e) a 256×256, lo que cuadruplica las multiplicaciones por ciclo de cada unidad (Google Cloud, 2025a). El precio de esa eficiencia es la rigidez: el compilador XLA debe conocer las formas de los tensores, y las operaciones que no son matriciales se ejecutan en unidades vectoriales mucho más lentas (Jouppi et al., 2023).

**NPU.** El Neural Engine de Apple ocupa una fracción pequeña del SoC y comparte la memoria con la CPU y la GPU, lo que elimina las copias por un bus externo. Su objetivo es la inferencia en el dispositivo con un presupuesto de pocos vatios, no el entrenamiento (Apple, 2023). La M3 Ultra, del Mac Studio de 2025, une dos chips M3 Max y llega a 32 núcleos de Neural Engine con hasta 512 GB de memoria unificada (Apple, 2025).

**FPGA.** La Alveo U55C no trae unidades de cómputo fijas. El diseñador sintetiza el circuito exacto que necesita la red, con la cantidad de multiplicadores DSP y la precisión de bits que requiera cada capa. Como el circuito se ejecuta sin sistema operativo ni planificador, la latencia de cada inferencia es determinística. A cambio, el desarrollo con herramientas HLS o con lenguajes de descripción de hardware exige meses de ingeniería especializada (AMD, 2023).

## 2. Benchmarks MLPerf Inference v4.1

MLPerf Inference mide el rendimiento con reglas comunes. En el escenario *Offline* se envían todas las muestras de una vez y se mide el throughput; en el escenario *Server* las consultas llegan siguiendo una distribución de Poisson y el sistema debe cumplir un límite de latencia en el percentil 99 (Reddi et al., 2020). Las siguientes tablas reúnen resultados de la división cerrada de v4.1, normalizados por acelerador dividiendo el resultado del sistema entre el número de aceleradores declarado en su archivo de sistema.

### 2.1 ResNet-50 v1.5 (clasificación de imágenes, Offline)

| Sistema (remitente) | Aceleradores | Muestras/s del sistema | Muestras/s por acelerador |
|---|---|---|---|
| 8× H200 SXM (NVIDIA) | 8 | 756 960 | 94 620 |
| 8× H100 SXM (Supermicro SYS-421GE) | 8 | 710 521 | 88 815 |
| 2× speedAI240 Preview (UntetherAI) | 2 | 140 625 | 70 313 |
| 6× speedAI240 Slim (UntetherAI) | 6 | 334 462 | 55 744 |
| 2× L40S (Lenovo) | 2 | 86 305 | 43 152 |
| 2× L4 (Lenovo) | 2 | 25 600 | 12 800 |
| 2× Xeon 6 Granite Rapids, solo CPU (Intel) | 2 CPU | 45 617 | — |

Fuente: MLCommons (2024).

### 2.2 BERT-Large, SQuAD v1.1 (bert-99)

| Sistema | Offline, muestras/s | Server, muestras/s | Latencia p99 en Server |
|---|---|---|---|
| 8× H200 SXM (NVIDIA) | 73 310 | 57 609 | 80,8 ms |
| 8× H100 SXM (Dell XE9680) | 70 594 | 56 012 | 76,3 ms |
| 2× Xeon 6 Granite Rapids (Intel) | 3 024 | 2 437 | 129,1 ms |

El límite de BERT en el escenario Server es de 130 ms en el percentil 99, así que el sistema de CPU opera justo en el borde mientras las GPU tienen margen. En el escenario de borde *SingleStream*, que mide una consulta a la vez, el percentil 90 de BERT es de 1,12 ms en la L40S y de 2,14 ms en la L4 (MLCommons, 2024).

### 2.3 Stable Diffusion XL (generación de imágenes, Offline)

| Sistema | Aceleradores | Muestras/s del sistema | Muestras/s por acelerador |
|---|---|---|---|
| 8× H200 SXM (NVIDIA) | 8 | 17,42 | 2,18 |
| 8× H100 SXM (NVIDIA DGX H100) | 8 | 16,35 | 2,04 |
| 1× GH200 Grace Hopper (NVIDIA) | 1 | 2,30 | 2,30 |
| 4× TPU v6e Trillium (Google) | 4 | 5,44 | 1,36 |
| 4× TPU v5e (Google) | 4 | 1,75 | 0,44 |
| 1× Jetson AGX Orin (NVIDIA) | 1 | 0,10 | 0,10 |

Fuente: MLCommons (2024).

### 2.4 Quién lidera y por qué

En **throughput**, la familia Hopper lidera los tres modelos. La H200 supera a la H100 en un 7 % en ResNet-50 y en un 7 % en SDXL, una diferencia que se explica por su memoria (141 GB de HBM3e a 4,8 TB/s frente a 80 GB de HBM3 a 3,35 TB/s) y no por más cómputo, porque ambos chips tienen los mismos SM (Genkina, 2024). En SDXL, Trillium triplica a la TPU v5e (1,36 frente a 0,44 muestras/s por chip), lo que coincide con la MXU cuatro veces mayor, pero todavía rinde un 33 % menos que una H100. La GPU conserva la ventaja porque la difusión combina convoluciones, atención y operaciones elementales que TensorRT fusiona en kernels específicos.

En **eficiencia energética**, solo NVIDIA y UntetherAI publicaron mediciones de potencia en v4.1 (Genkina, 2024). En ResNet-50 Server, el sistema de 8× H200 en modo MaxQ alcanzó 480 131 muestras/s con 5013,79 W, es decir, 95,8 muestras/s/W, mientras que el sistema de 6× speedAI240 Slim alcanzó 309 752 muestras/s con 985,52 W, es decir, 314,3 muestras/s/W: 3,3 veces más eficiente (Genkina, 2024). La razón es arquitectónica: el speedAI240 coloca el cómputo junto a la SRAM (*at-memory compute*) y evita mover datos a la DRAM, que es donde se gasta la mayor parte de la energía. En BERT y SDXL no hubo envíos de potencia de otros fabricantes; como referencia, el promedio del registro de potencia `spl.txt` del sistema H200 MaxQ da unos 5230 W en BERT Offline (≈10 muestras/s/W) y unos 4490 W en SDXL Offline (≈0,003 muestras/s/W). Estas dos cifras son un cálculo propio sobre el registro completo, no un resultado oficial.

Las NPU de teléfonos y la Neural Engine de Apple no participan en MLPerf Inference de centro de datos ni de borde, por lo que no existen cifras comparables con esta tabla. Tampoco hubo envíos de FPGA en la división cerrada de v4.1: los 92 sistemas declarados usan GPU de NVIDIA o AMD, TPU de Google, el speedAI240 de UntetherAI o solo CPU. Esa ausencia es en sí un dato: estas plataformas compiten en nichos donde el throughput agregado no es la métrica decisiva.
