-----------------------------------------
      Step   CPU (ms)   GPU (ms)  Speedup
-----------------------------------------
   phantom        3.7        2.5     1.5x
   forward     1025.6     6848.6     0.1x
      reco    22788.9     3352.9     6.8x
    interp       93.3       59.8     1.6x
     total    23913.2    10265.6     2.3x
-----------------------------------------

Overall speedup: 2.3x


-----------------------------------------
      Step   CPU (ms)   GPU (ms)  Speedup
-----------------------------------------
   phantom        3.0        2.5     1.2x
   forward      312.6      324.8     1.0x
      reco    21783.2     4077.4     5.3x
    interp       60.5       75.1     0.8x
     total    22159.6     4480.7     4.9x
-----------------------------------------

Overall speedup: 4.9x


-----------------------------------------
    Step   CPU (ms)   GPU (ms)  Speedup
-----------------------------------------
phantom        2.5        3.2     0.8x
forward      333.9      348.1     1.0x
    reco    22800.9     3230.8     7.1x
    interp       68.8       63.0     1.1x
    total    23210.3     3646.0     6.4x
-----------------------------------------

Overall speedup: 6.4x



     -----------------------------------------
           Step   CPU (ms)   GPU (ms)  Speedup
     -----------------------------------------
        phantom        3.2        3.3     1.0x
        forward      349.1      361.5     1.0x
           reco       31.6       40.1     0.8x
         interp       72.5       68.8     1.1x
          total      458.1      474.8     1.0x
     -----------------------------------------

     Overall speedup: 1.0x


  ⎿  GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
     Benchmark: 40 samples, level 1, mode=full

     Running CPU benchmark...
       CPU total: 24.0s (194 ms/sample avg)
     Running GPU benchmark...
       GPU total: 11.6s (168 ms/sample avg)

     -----------------------------------------
           Step   CPU (ms)   GPU (ms)  Speedup
     -----------------------------------------
        phantom        2.9        2.8     1.0x
        forward      160.8      160.9     1.0x
          noise        0.2        0.0     8.2x
           reco       28.7        2.4    11.8x
         interp        0.6        0.8     0.7x
          total      194.1      168.0     1.2x
     -----------------------------------------

     Overall speedup: 1.2x

     Running multiprocess benchmark (4 workers, 40 samples total)...
       Wall time: 207.3s (5183 ms/sample throughput)
       vs serial CPU: 0.0x throughput gain
     Active optimizations: A, B, C, D, E, F, G, H
     Results saved to: results\gpu_benchmark_5.json