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

## 3. Análisis costo-rendimiento y TCO

El TCO a tres años se aproxima como costo de adquisición más energía más ingeniería de software. Para la energía se supone una tarifa de 0,12 USD/kWh y un PUE de 1,5 en centro de datos; ambos valores son supuestos de este análisis y se declaran para que el cálculo sea reproducible. Tres años equivalen a 26 280 horas.

| Opción | Adquisición o alquiler (3 años) | Energía (3 años) | Ingeniería de software | Observación |
|---|---|---|---|---|
| 1× H100 SXM comprada | ≈30 000 USD (precio de referencia de la guía) | 0,7 kW × 26 280 h × 1,5 × 0,12 ≈ 3311 USD | Baja: TensorRT y Triton | Más el servidor anfitrión |
| 1× TPU v5e alquilada | 1,20 USD/h bajo demanda ≈ 31 536 USD; 0,54 USD/h con compromiso de 3 años ≈ 14 191 USD | Incluida | Media: adaptar a JAX/XLA | Solo en Google Cloud (Google Cloud, 2025b) |
| 1× TPU v6e alquilada | 2,70 USD/h ≈ 70 956 USD; 1,22 USD/h con compromiso ≈ 32 062 USD | Incluida | Media | Solo en Google Cloud (Google Cloud, 2025b) |
| 1× Alveo U55C | 4795 USD de lanzamiento (HPCwire, 2021) a ≈8000 USD (referencia de la guía) | 0,15 kW × 26 280 h × 1,5 × 0,12 ≈ 710 USD | Alta: meses de diseño HLS | Latencia determinística |
| Mac Studio M3 Ultra | Desde 3999 USD (Apple, 2025) | Bajo | Media: Core ML o MLX | No es un producto de servidor |

**Escenario (a): startup con inferencia en la nube de menos de 100 solicitudes por segundo.** Una sola H100 atiende unas 7000 consultas BERT por segundo en el escenario Server (56 012 / 8), así que 100 solicitudes por segundo usan menos del 2 % de su capacidad. Comprar una GPU de 30 000 USD para usarla al 2 % es la peor opción de costo. Lo racional es alquilar la unidad más pequeña disponible y pagar solo lo que se usa: una TPU v5e con compromiso cuesta unos 14 000 USD en tres años con energía incluida, y una instancia de GPU pequeña en la nube, del tipo L4, ofrece el mismo ecosistema CUDA que la empresa usaría después para escalar. La recomendación para este escenario es la nube, con TPU v5e si el modelo ya está en JAX o TensorFlow, y con GPU en la nube si el equipo trabaja en PyTorch y necesita operaciones personalizadas. La FPGA se descarta porque su ingeniería no se amortiza a este volumen.

**Escenario (b): inferencia en el borde.** En un teléfono, la NPU ya viene pagada dentro del SoC y opera con pocos vatios; ninguna otra opción compite en costo marginal. En un vehículo o en una cámara industrial, las alternativas son un módulo de tipo Jetson AGX Orin, una tarjeta de bajo consumo como la L4 o el speedAI240 Slim de 75 W, o una FPGA. Para el borde, el costo dominante a tres años no es la energía sino la cantidad de unidades: 50 unidades de 8000 USD suman 400 000 USD, mientras que la energía de cada FPGA ronda los 710 USD en el mismo periodo. Por eso el borde favorece la NPU cuando el dispositivo ya la trae, y favorece la FPGA o la tarjeta de bajo consumo solo cuando un requisito de latencia o de fiabilidad lo justifica.

## 4. Recomendación fundamentada por perfil

**Perfil A: laboratorio universitario con 10 000 USD, visión y NLP de hasta 7B parámetros.** La recomendación es una estación de trabajo con una GPU NVIDIA de 24 GB, del tipo RTX 4090, cuyo precio de lanzamiento fue de 1599 USD (NVIDIA, 2022b). Los criterios son tres. Primero, la capacidad: un modelo de 7B ocupa unos 14 GB en FP16, así que cabe completo para inferencia, y se puede ajustar con LoRA o QLoRA. Segundo, el ecosistema: PyTorch, cuDNN y la mayor parte del código de investigación publicado asumen CUDA. Tercero, el costo: el presupuesto alcanza para la estación completa e incluso para una segunda GPU. La limitación es que no permite entrenar un modelo de 7B desde cero ni ajustar todos sus parámetros en FP16. Un Mac Studio M3 Ultra, con 96 GB de memoria unificada desde 3999 USD (Apple, 2025), es la alternativa si se prioriza cargar modelos grandes sin cuantizar, a costa de un ecosistema de investigación más pequeño.

**Perfil B: inspección visual en tiempo real con menos de 5 ms en 50 líneas.** La recomendación es una FPGA por línea. Los criterios son la latencia máxima determinística, la ausencia de un punto único de falla y la cercanía física a la cámara. La limitación es el costo de capital y de desarrollo. El razonamiento completo está en la Decisión 3.

**Perfil C: proveedor de nube que entrena modelos de 70B en clústeres de 512 aceleradores.** La recomendación es un clúster de GPU H100 o H200 con NVLink dentro de cada nodo e InfiniBand entre nodos. Los criterios son tres. Primero, la comunicación: el entrenamiento de 70B usa paralelismo de tensores, de tuberías y de datos, y NVLink ofrece 900 GB/s por GPU dentro del nodo, mientras que NCCL implementa las operaciones colectivas sobre la red (NVIDIA, 2022a). Segundo, la disponibilidad: un proveedor de nube distinto de Google no puede comprar TPU, que solo se ofrecen como servicio de Google Cloud. Tercero, la demanda de sus clientes, que en su mayoría usan PyTorch sobre CUDA. La limitación es el consumo: 512 GPU de 700 W suman 358 kW solo en aceleradores. Si el proveedor fuera Google, los pods de TPU con conmutadores ópticos serían una alternativa más eficiente: Jouppi et al. (2023) reportan que la TPU v4 rinde entre 1,2 y 1,7 veces más que la A100 con 1,3 a 1,9 veces menos energía.

## 5. Cuantización INT8 frente a FP32

Pasar de FP16 a INT8 duplica las operaciones por ciclo de un Tensor Core, porque cada operando ocupa la mitad de bits. Ese patrón se repite en varias generaciones de GPU, como muestran las hojas de especificaciones:

| GPU | FP16 en Tensor Core | INT8 en Tensor Core | Razón | Fuente |
|---|---|---|---|---|
| T4 (Turing) | 65 TFLOPS | 130 TOPS | 2× | NVIDIA (2019) |
| A100 (Ampere) | 312 TFLOPS | 624 TOPS | 2× | NVIDIA (2020) |
| H100 SXM (Hopper) | 989 TFLOPS | 1979 TOPS | 2× | NVIDIA (2022a) |

La exactitud no se duplica, pero se conserva si la cuantización se calibra:

| Modelo | FP32 | INT8 con calibración | INT8 mal calibrado | Requisito MLPerf |
|---|---|---|---|---|
| ResNet-50 v1.5 (top-1) | 76,16 % | 76,05 % (entropía) | 75,82 % (máximo) | ≥ 99 % de 76,46 % = 75,70 % |
| BERT-Large SQuAD v1.1 (F1) | 91,01 | 90,20 (percentil 99,999 %); 90,67 con QAT | 37,40 (entropía) | ≥ 99 % o ≥ 99,9 % de 90,874 |

Fuentes: Wu et al. (2020) para la exactitud; MLCommons (2024) y Reddi et al. (2020) para los requisitos.

El caso de BERT muestra que la cuantización no es un simple cambio de tipo. Con calibración por entropía, el F1 cae de 91,01 a 37,40, porque las activaciones de la función GELU tienen valores atípicos muy grandes y el método elige un rango que los recorta. Con el percentil 99,999 % el F1 baja solo a 90,20, y con entrenamiento consciente de la cuantización (QAT) llega a 90,67 (Wu et al., 2020). MLPerf exige que cada envío cuantizado conserve al menos el 99 % de la exactitud FP32 de referencia, o el 99,9 % en las variantes más estrictas de BERT, medido sobre todo el conjunto de validación. Por eso los envíos INT8 de la Sección 2 son comparables con los de mayor precisión.

## 6. Decisiones técnicas sobre el Perfil B

### Decisión 3: ¿GPU centralizada o FPGA distribuida para el Perfil B?

En throughput, una sola GPU alcanza con holgura. Cincuenta líneas a 10 imágenes por segundo producen 500 imágenes por segundo. Con TensorRT, un modelo del tamaño de ResNet-18 tarda entre 1 y 2 ms por imagen con batch 1, y como la H100 ejecuta varias inferencias en paralelo, su capacidad real es mucho mayor que 500 por segundo: en MLPerf v4.1, cada H200 atiende unas 79 000 consultas ResNet-50 por segundo en el escenario Server (632 229 / 8), con un modelo más pesado que ResNet-18 (MLCommons, 2024). Con una utilización inferior al 1 %, la espera promedio en la cola es prácticamente nula, así que la latencia promedio queda muy por debajo de 5 ms.

En costo de capital, la GPU también gana: un servidor con una H100 cuesta unos 30 000 USD, frente a 50 × 8000 = 400 000 USD de las FPGA, más de 13 veces menos. Pero la centralización introduce un punto único de falla: si el servidor, la GPU o la red caen, las 50 líneas se quedan sin inspección al mismo tiempo. Además, la latencia promedio no es la latencia máxima. La GPU comparte el dispositivo entre 50 flujos y depende del sistema operativo, del controlador y del planificador de CUDA, así que un pico de carga o una pausa del sistema puede producir una imagen de 8 o 10 ms. La FPGA no tiene esas capas: su circuito tarda siempre lo mismo.

Como el Perfil B rechaza piezas defectuosas y tiene implicaciones de seguridad industrial, el requisito real es una latencia máxima garantizada, no un promedio. Por eso se recomienda la FPGA distribuida, aunque cueste más. Si la inspección fuera solo de control de calidad, sin riesgo para la seguridad, la GPU centralizada sería razonable, siempre con un segundo servidor de respaldo, lo que llevaría su costo a unos 60 000 USD, todavía muy por debajo del de las FPGA.

### Decisión 4: ¿conviene cuantizar a INT8 el modelo del Perfil B?

Si el modelo ya cumple los 5 ms en FP16, INT8 no hace falta para cumplir el requisito. Su beneficio es el margen: duplicar el throughput del mismo hardware permite atender el doble de líneas o de imágenes por segundo sin comprar más aceleradores, y deja holgura para absorber picos de carga sin acercarse al límite de latencia. En una FPGA, además, los multiplicadores INT8 ocupan menos slices DSP, lo que libera área para un modelo más grande o reduce el chip necesario.

El riesgo está en cómo se valida la exactitud. Una caída de menos de un punto en el top-1 promedio no dice nada sobre la clase "defecto", que probablemente es minoritaria en los datos. Las clases minoritarias tienen menos ejemplos con los que ajustar la calibración, así que su exactitud puede caer mucho más que el promedio sin que el top-1 general lo muestre. El caso de BERT, cuyo F1 cae de 91,01 a 37,40 con un método de calibración inadecuado, muestra que la cuantización puede fallar de forma severa cuando el rango de las activaciones no se elige bien (Wu et al., 2020). En este dominio, un falso negativo deja pasar una pieza defectuosa, mientras que un falso positivo solo envía una pieza buena a revisión manual; el costo de los dos errores no es simétrico.

Por eso se recomienda cuantizar a INT8 con una condición: antes de desplegarlo, el equipo debe medir el recall y la precisión de la clase "defecto" sobre un conjunto de validación propio, y comparar ambos contra el modelo FP16. Si el recall de los defectos no cae de forma significativa, el margen ganado justifica el cambio. Si cae, conviene probar otra calibración o QAT, o mantener FP16, porque el throughput extra no compensa piezas defectuosas no detectadas.

## Conclusiones

No existe un acelerador mejor en términos absolutos. Los resultados de MLPerf v4.1 muestran que la GPU Hopper lidera en throughput en los tres modelos analizados y es la opción más general, mientras que la TPU Trillium triplicó a su predecesora en SDXL y sigue un 33 % por debajo de la H100 por chip. En eficiencia energética, el speedAI240 de UntetherAI fue 3,3 veces más eficiente que la H200 en ResNet-50, lo que confirma que mover datos cuesta más energía que calcular. El costo cambia la recomendación según la escala: con poca carga conviene alquilar en la nube, en el borde conviene la NPU que ya trae el dispositivo, y la FPGA se justifica cuando la latencia máxima debe estar garantizada. La cuantización INT8 duplica el throughput de forma predecible, pero solo es segura si se calibra y se valida sobre la clase que más importa detectar.

## Referencias

AMD. (2023). *Alveo U55C data center accelerator card data sheet (DS978, v1.3)*. Advanced Micro Devices. https://docs.amd.com/r/en-US/ds978-u55c

Apple. (2023, 30 de octubre). *Apple unveils M3, M3 Pro, and M3 Max, the most advanced chips for a personal computer* [Comunicado de prensa]. https://www.apple.com/newsroom/

Apple. (2025, 5 de marzo). *Apple unveils new Mac Studio, the most powerful Mac ever* [Comunicado de prensa]. https://www.apple.com/newsroom/

Genkina, D. (2024, 28 de agosto). AI inference competition heats up: First MLPerf benchmarks for Nvidia Blackwell, AMD, Google, Untether AI. *IEEE Spectrum*. https://spectrum.ieee.org/new-inference-chips

Google Cloud. (2025a). *TPU v6e*. Documentación de Cloud TPU. https://docs.cloud.google.com/tpu/docs/v6e

Google Cloud. (2025b). *Cloud TPU pricing*. https://cloud.google.com/tpu/pricing

HPCwire. (2021, 15 de noviembre). *Xilinx targets HPC and datacenter with new Alveo U55C FPGA card*. https://www.hpcwire.com/2021/11/15/xilinx-targets-hpc-and-datacenter-with-new-alveo-u55c-fpga-card/

Jouppi, N. P., Kurian, G., Li, S., Ma, P., Nagarajan, R., Nai, L., Patil, N., Subramanian, S., Swing, A., Towles, B., Young, C., Zhou, X., Zhou, Z., & Patterson, D. (2023). TPU v4: An optically reconfigurable supercomputer for machine learning with hardware support for embeddings. En *Proceedings of the 50th Annual International Symposium on Computer Architecture* (pp. 1–14). ACM. https://doi.org/10.1145/3579371.3589350

MLCommons. (2024). *MLPerf Inference v4.1 results* [Conjunto de datos]. https://github.com/mlcommons/inference_results_v4.1

NVIDIA. (2019). *NVIDIA T4 Tensor Core GPU datasheet*. NVIDIA Corporation.

NVIDIA. (2020). *NVIDIA A100 Tensor Core GPU architecture* [Libro blanco]. NVIDIA Corporation.

NVIDIA. (2022a). *NVIDIA H100 Tensor Core GPU architecture* [Libro blanco]. NVIDIA Corporation.

NVIDIA. (2022b, 20 de septiembre). *NVIDIA delivers quantum leap in performance, introduces new era of neural rendering with GeForce RTX 40 Series* [Comunicado de prensa]. https://nvidianews.nvidia.com/

NVIDIA. (2023). *NVIDIA H100 Tensor Core GPU datasheet*. NVIDIA Corporation.

Reddi, V. J., Cheng, C., Kanter, D., Mattson, P., Schmuelling, G., Wu, C.-J., et al. (2020). MLPerf inference benchmark. En *Proceedings of the ACM/IEEE 47th Annual International Symposium on Computer Architecture* (pp. 446–459). IEEE. https://doi.org/10.1109/ISCA45697.2020.00045

Wu, H., Judd, P., Zhang, X., Isaev, M., & Micikevicius, P. (2020). *Integer quantization for deep learning inference: Principles and empirical evaluation* (arXiv:2004.09602). arXiv. https://arxiv.org/abs/2004.09602
