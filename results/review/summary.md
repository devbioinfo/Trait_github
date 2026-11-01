# Review checks: summary

## 1. Development versus held-out control traits

```
coding                trait           controls  n any_loss_call  mean_false_loss_calls
  L+UL               marine development (1-20) 20   20% (6-44%)                  0.250
  L+UL               marine  held out (21-100) 80   15% (8-25%)                  0.188
  L+UL hypoxia_subterranean development (1-20) 20   10% (1-32%)                  0.100
  L+UL hypoxia_subterranean  held out (21-100) 80    6% (2-14%)                  0.075
  L+UL     hypoxia_altitude development (1-20) 20   10% (1-32%)                  0.100
  L+UL     hypoxia_altitude  held out (21-100) 80  21% (13-32%)                  0.800
  L+UL              hypoxia development (1-20) 20   10% (1-32%)                  0.100
  L+UL              hypoxia  held out (21-100) 80   10% (4-19%)                  0.125
  L+UL         echolocation development (1-20) 20   15% (3-38%)                  0.150
  L+UL         echolocation  held out (21-100) 80   11% (5-20%)                  0.238
L only               marine development (1-20) 20    5% (0-25%)                  0.050
L only               marine  held out (21-100) 80    6% (2-14%)                  0.162
L only hypoxia_subterranean development (1-20) 20   15% (3-38%)                  0.150
L only hypoxia_subterranean  held out (21-100) 80   10% (4-19%)                  0.175
L only     hypoxia_altitude development (1-20) 20   20% (6-44%)                  0.200
L only     hypoxia_altitude  held out (21-100) 80   15% (8-25%)                  0.238
L only              hypoxia development (1-20) 20   10% (1-32%)                  0.100
L only              hypoxia  held out (21-100) 80   15% (8-25%)                  0.212
L only         echolocation development (1-20) 20    5% (0-25%)                  0.050
L only         echolocation  held out (21-100) 80    4% (1-11%)                  0.038
```

## 2. Composite-null tier (100,000 replicates per gene and origin)

```
               trait  loss_tested  global_calls  composite_null_calls             genes  min_iut_p
              marine        12851            48                     0                      0.00004
hypoxia_subterranean        12851            15                     3 CRYGS, LGSN, MPP4    0.00001
```

## 3. Call-set stability

```
                                 run  calls  main_calls  shared  jaccard                                                                           lost_from_main                                                                                           new  controls_any_loss_call
               ULintact_echolocation      7           0       0     0.00                                                                                                                                    HSD17B13, METTL22, OR5K3, OR6P1, PLAC8, TSTD1, YJU2                    0.05
                    ULintact_hypoxia      6          11       5     0.42                                                   ADH7, ANKRD13D, GJA8, LCTL, MPP4, ROM1                                                                                        B3GNT4                    0.25
           ULintact_hypoxia_altitude      1           1       0     0.00                                                                                    DEAF1                                                                                         AUTS2                    0.05
       ULintact_hypoxia_subterranean     17          15       9     0.39                                                   CABP5, CRYBA4, GJA8, MPP4, OPTC, RGS11                                  ABCA4, ANKRD33, CRYBA1, CRYBB1, DDIT4L, IMPG1, PPM1N, ZNF862                    0.05
                     ULintact_marine      7          48       3     0.06 ACSM4, ADGRF2, ANXA9, ATP13A5, CALHM1, CCDC170, GBA3, GLYATL3, GNAT3, GSDMA, HTRA4, KLK8                                                                BPIFA2, KCNMB3, KRT77, OR52P1P                    0.25
         cap0.2_hypoxia_subterranean     17          15      14     0.78                                                                                   B3GNT4                                                                          ADH7, OR10A7, ZNF862                     NaN
                       cap0.2_marine     65          48      47     0.71                                                                                     MYH6         AZGP1, BPIFA2, FMO1, HLA-E, KLK12, KRT82, OR2A14, OR2T6, OR4Q3, OR51B6, OR52A5, OR6K2                     NaN
         cap0.9_hypoxia_subterranean     15          15      14     0.88                                                                                  TPD52L3                                                                                        ZNF862                     NaN
                       cap0.9_marine     48          48      46     0.92                                                                              MYH6, OR6C2                                                                                 B3GNT3, OR6K2                     NaN
falseintact0.01_hypoxia_subterranean     15          15      15     1.00                                                                                                                                                                                                            NaN
              falseintact0.01_marine     49          48      47     0.94                                                                                   CALHM1                                                                                  BPIFA2, FMO1                     NaN
      shrink0.1_hypoxia_subterranean     16          15      15     0.94                                                                                                                                                                                 OR10A7                     NaN
                    shrink0.1_marine     62          48      47     0.75                                                                                    ANXA9  ADGRF3, BPIFA2, C16orf89, CRYZ, FMO1, KIAA0513, MRPS9, OR51B6, OR52K1, OR56B2P, OR6K2, RPL26                     NaN
      shrink0.3_hypoxia_subterranean     13          15      12     0.75                                                                       CRYBA4, GJA8, OPTC                                                                                        CRYBA1                     NaN
                    shrink0.3_marine     55          48      37     0.56   ADGRF2, ANXA9, KRT1, KRTAP24-1, MRGPRX2, MSS51, OR2K2, OR5H14, SERPINB12, SLC4A9, TLE7 ADGRF3, BPIFA2, C16orf89, CRYZ, FMO1, KCNMB3, KIAA0513, MRPS9, OR4Q3, OR51B6, OR52K1, OR56B2P                     NaN
```

## 4. Enrichment with and without collapsing gene families

```
  source  collapsed                trait  significant_terms                                                                                                                                                                                                 top
  tissue      False hypoxia_subterranean                  1                                                                                                                                                                                              retina
  tissue      False               marine                  0                                                                                                                                                                                                    
  tissue       True hypoxia_subterranean                  1                                                                                                                                                                                              retina
  tissue       True               marine                  0                                                                                                                                                                                                    
reactome      False               marine                  5 Sensory Perception, Expression and translocation of olfactory receptors, Olfactory Signaling Pathway, Sensory perception of taste, Sensory perception of sweet, bitter, and umami (glutamate) taste
reactome       True               marine                  5 Sensory Perception, Sensory perception of taste, Sensory perception of sweet, bitter, and umami (glutamate) taste, Expression and translocation of olfactory receptors, Olfactory Signaling Pathway
      go      False hypoxia_subterranean                  7                                                          lens development in camera-type eye, camera-type eye development, visual perception, sensory perception of light stimulus, eye development
      go      False               marine                 13             sensory perception of chemical stimulus, nervous system process, sensory perception, detection of stimulus involved in sensory perception, G protein-coupled receptor signaling pathway
      go       True hypoxia_subterranean                  7                                                          lens development in camera-type eye, camera-type eye development, visual perception, sensory perception of light stimulus, eye development
      go       True               marine                 12             sensory perception of chemical stimulus, nervous system process, sensory perception, detection of stimulus involved in sensory perception, G protein-coupled receptor signaling pathway
```
