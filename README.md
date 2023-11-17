# HighPerformanceComputing
The repo for the High Performance Computing Class assignments

# Profiling - Performance Results
* perf
  ```
   Performance counter stats for './2mm_acc':
  
        95227,811486      task-clock (msec)         #    1,000 CPUs utilized          
                 771      context-switches          #    0,008 K/sec                  
                   1      cpu-migrations            #    0,000 K/sec                  
               2.640      page-faults               #    0,028 K/sec                  
     140.822.718.715      cycles                    #    1,479 GHz                    
      18.525.500.537      instructions              #    0,13  insn per cycle         
     <not supported>      branches                                                    
           2.895.561      branch-misses                                               
  
        95,255688796 seconds time elapsed     
       6.771.889.947      cache-references                                            
       1.305.708.970      cache-misses              #   19,281 % of all cache refs    
  
       111,895854170 seconds time elapsed"
  ```
* valgrind
  ```
  I     refs:      18,302,123,533
  ==12529== I1  misses:             1,295
  ==12529== LLi misses:             1,100
  ==12529== I1  miss rate:           0.00%
  ==12529== LLi miss rate:           0.00%
  ==12529== 
  ==12529== D   refs:       6,449,878,101  (4,296,082,343 rd   + 2,153,795,758 wr)
  ==12529== D1  misses:     2,153,243,374  (2,152,587,160 rd   +       656,214 wr)
  ==12529== LLd misses:     2,150,876,421  (2,150,220,326 rd   +       656,095 wr)
  ==12529== D1  miss rate:           33.4% (         50.1%     +           0.0%  )
  ==12529== LLd miss rate:           33.3% (         50.1%     +           0.0%  )
  ==12529== 
  ==12529== LL refs:        2,153,244,669  (2,152,588,455 rd   +       656,214 wr)
  ==12529== LL misses:      2,150,877,521  (2,150,221,426 rd   +       656,095 wr)
  ==12529== LL miss rate:             8.7% (          9.5%     +           0.0%  )"
  ```
# Parallelization - Performance Results
## Improved only the `kernel_2mm` function
| Time exec | Improvement approach |
| --- | --- |
| 94.811802728 seconds | no improvements - original code |
| 41.349425178 seconds | 2 parallel & 2 for |
| 39.626148795 seconds | 1 parallel & 2 for |
| 38.969717257 seconds | parallel for with 4 threads |

## Improved the `kernel_2mm` and the `init_array` functions
| Time exec | Improvement approach |
| --- | --- |
| 94.811802728 seconds | no improvements - original code |
| 57.400443969 seconds | parallel sections |
| 37.871657149 seconds | more parallel & more for |
| 37.380195890 seconds | 1 parallel & more for |
| 44.245200038 seconds | parallel for with 4 threads for each "for couple" |

## OFFLOADING on `kernel_2mm`
| Time exec | Improvement approach |
| --- | --- |
| 34.238667050 seconds | with target parallel for map |

## OFFLOADING on `kernel_2mm` and `init_array`
| Time exec | Improvement approach |
| --- | --- |
| 33.478696708 seconds | with target parallel for map |

## MIX - OFFLOADING on `kernel_2mm` & Parallelization on `init_array`
| Time exec | Improvement approach |
| --- | --- |
| 31.766432255 seconds | mix |
| 35.203977347 seconds | mix with TEAM |


# Authors
Goldoni Lorena

Panini Gabriel

Taoufiq Adam

