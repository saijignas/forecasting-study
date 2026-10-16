# Results tables

Generated from forecasts produced at commit `4ad1298f3a81c46a88642e285dfdcf3693047218` (see results/run_meta.json).

## Hypothesis decisions

### monthly (confirmatory, Holm-adjusted)

| Hypothesis | Point criteria | p+ (raw) | p- (raw) | p+ adj | p- adj | Verdict |
|---|---|---|---|---|---|---|
| H1 | 0/1 true | 0.9993 | 0.0029 | 1.0000 | 0.0116 | not supported |
| H2 | 0/1 true | 0.9509 | 0.1009 | 1.0000 | 0.2018 | inconclusive |
| H3 | 6/10 true | 0.9983 | 0.0068 | 1.0000 | 0.0204 | not supported |
| H4 | 1/1 true | 0.3819 | 1.0000 | 1.0000 | 1.0000 | inconclusive (point estimates meet the criterion but the claim is not statistically established) |

### daily (secondary replication, no Holm)

| Hypothesis | Point criteria | p+ (raw) | p- (raw) | p+ adj | p- adj | Verdict |
|---|---|---|---|---|---|---|
| H1 | 1/1 true | 0.5745 | 1.0000 | 0.5745 | 1.0000 | inconclusive (point estimates meet the criterion but the claim is not statistically established) |
| H2 | 1/1 true | 0.4182 | 1.0000 | 0.4182 | 1.0000 | inconclusive (point estimates meet the criterion but the claim is not statistically established) |
| H3 | 6/10 true | 1.0000 | 0.0000 | 1.0000 | 0.0000 | not supported |

## monthly

### Error by horizon

| model | h | MASE_mean | MASE_median | sMAPE_mean | RMSSE_mean |
|---|---|---|---|---|---|
| NAIVE | 1 | 0.702 | 0.410 | 3.068 | 0.574 |
| NAIVE | 3 | 1.502 | 0.954 | 6.084 | 1.220 |
| NAIVE | 6 | 1.336 | 1.235 | 5.817 | 1.117 |
| NAIVE | 12 | 1.372 | 1.218 | 5.492 | 1.238 |
| SNAIVE | 1 | 2.258 | 2.237 | 11.276 | 2.327 |
| SNAIVE | 3 | 1.830 | 1.727 | 8.093 | 1.643 |
| SNAIVE | 6 | 1.530 | 1.454 | 6.569 | 1.349 |
| SNAIVE | 12 | 1.372 | 1.218 | 5.492 | 1.238 |
| DRIFT | 1 | 0.690 | 0.425 | 3.044 | 0.563 |
| DRIFT | 3 | 1.467 | 0.904 | 6.026 | 1.181 |
| DRIFT | 6 | 1.258 | 1.096 | 5.725 | 1.048 |
| DRIFT | 12 | 1.117 | 1.099 | 5.018 | 1.055 |
| SMEAN | 1 | 3.713 | 3.083 | 14.558 | 2.845 |
| SMEAN | 3 | 3.487 | 2.358 | 12.866 | 2.669 |
| SMEAN | 6 | 3.456 | 2.250 | 12.705 | 2.631 |
| SMEAN | 12 | 3.405 | 2.376 | 11.786 | 2.637 |
| ETS | 1 | 0.643 | 0.433 | 3.292 | 0.580 |
| ETS | 3 | 0.743 | 0.689 | 3.840 | 0.640 |
| ETS | 6 | 0.904 | 0.947 | 4.786 | 0.817 |
| ETS | 12 | 1.153 | 0.998 | 5.284 | 1.070 |
| ARIMA | 1 | 0.640 | 0.478 | 3.263 | 0.597 |
| ARIMA | 3 | 0.688 | 0.697 | 3.388 | 0.612 |
| ARIMA | 6 | 0.914 | 0.974 | 4.412 | 0.837 |
| ARIMA | 12 | 1.148 | 1.166 | 5.239 | 1.086 |
| THETA | 1 | 0.492 | 0.368 | 2.260 | 0.415 |
| THETA | 3 | 0.853 | 0.711 | 3.781 | 0.710 |
| THETA | 6 | 0.950 | 0.943 | 4.273 | 0.854 |
| THETA | 12 | 1.236 | 1.184 | 5.179 | 1.148 |
| ML-G | 1 | 0.598 | 0.555 | 3.002 | 0.572 |
| ML-G | 3 | 0.937 | 0.944 | 4.566 | 0.853 |
| ML-G | 6 | 0.975 | 0.920 | 5.015 | 0.892 |
| ML-G | 12 | 1.260 | 1.164 | 5.911 | 1.173 |
| ML-L | 1 | 0.724 | 0.629 | 3.629 | 0.686 |
| ML-L | 3 | 1.101 | 1.092 | 5.504 | 1.030 |
| ML-L | 6 | 1.268 | 1.282 | 6.336 | 1.168 |
| ML-L | 12 | 1.598 | 1.425 | 7.521 | 1.460 |
| COMB | 1 | 0.566 | 0.483 | 2.833 | 0.530 |
| COMB | 3 | 0.705 | 0.730 | 3.528 | 0.625 |
| COMB | 6 | 0.837 | 0.765 | 4.282 | 0.764 |
| COMB | 12 | 1.037 | 0.962 | 4.895 | 0.992 |

### Win rates against SNAIVE

|  | win_rate_margin0 | win_rate_margin0.03 | mean_MASE_ref_h | wilcoxon_p_secondary |
|---|---|---|---|---|
| ETS | 0.433 | 0.433 | 1.153 | 0.309 |
| ARIMA | 0.500 | 0.467 | 1.148 | 0.253 |
| THETA | 0.733 | 0.567 | 1.236 | 0.004 |
| ML-G | 0.333 | 0.333 | 1.260 | 0.685 |
| ML-L | 0.367 | 0.300 | 1.598 | 0.031 |
| COMB | 0.600 | 0.567 | 1.037 | 0.016 |
| STAT* | 0.467 | 0.467 | 1.124 | 0.146 |

### Intervals

| interval | level | model | coverage | ci95_lo | ci95_hi | mean_scaled_width | mean_pinball80 |
|---|---|---|---|---|---|---|---|
| conformal | 80 | NAIVE | 0.879 | 0.838 | 0.915 | 5.774 | 0.446 |
| conformal | 80 | SNAIVE | 0.832 | 0.783 | 0.876 | 6.607 | 0.630 |
| conformal | 80 | DRIFT | 0.881 | 0.839 | 0.918 | 5.593 | 0.420 |
| conformal | 80 | SMEAN | 0.703 | 0.629 | 0.769 | 8.570 | 1.028 |
| conformal | 80 | ETS | 0.877 | 0.845 | 0.906 | 4.368 | 0.329 |
| conformal | 80 | ARIMA | 0.880 | 0.849 | 0.908 | 4.755 | 0.335 |
| conformal | 80 | THETA | 0.870 | 0.823 | 0.911 | 4.245 | 0.324 |
| conformal | 80 | ML-G | 0.856 | 0.818 | 0.892 | 4.681 | 0.357 |
| conformal | 80 | ML-L | 0.849 | 0.805 | 0.886 | 5.048 | 0.435 |
| conformal | 80 | COMB | 0.873 | 0.840 | 0.903 | 4.305 | 0.310 |
| conformal | 95 | NAIVE | 0.952 | 0.918 | 0.979 | 11.484 |  |
| conformal | 95 | SNAIVE | 0.924 | 0.880 | 0.959 | 13.241 |  |
| conformal | 95 | DRIFT | 0.956 | 0.924 | 0.981 | 11.301 |  |
| conformal | 95 | SMEAN | 0.881 | 0.808 | 0.942 | 13.575 |  |
| conformal | 95 | ETS | 0.968 | 0.944 | 0.988 | 10.427 |  |
| conformal | 95 | ARIMA | 0.968 | 0.944 | 0.988 | 10.904 |  |
| conformal | 95 | THETA | 0.956 | 0.921 | 0.984 | 9.798 |  |
| conformal | 95 | ML-G | 0.963 | 0.936 | 0.983 | 10.185 |  |
| conformal | 95 | ML-L | 0.951 | 0.919 | 0.977 | 10.476 |  |
| conformal | 95 | COMB | 0.967 | 0.941 | 0.988 | 9.968 |  |
| model-based | 80 | ETS | 0.796 | 0.725 | 0.856 | 3.201 | 0.310 |
| model-based | 80 | ARIMA | 0.758 | 0.695 | 0.814 | 2.763 | 0.302 |
| model-based | 80 | THETA | 0.802 | 0.682 | 0.903 | 5.509 | 0.389 |
| model-based | 95 | ETS | 0.908 | 0.865 | 0.945 | 4.928 |  |
| model-based | 95 | ARIMA | 0.871 | 0.820 | 0.916 | 4.226 |  |
| model-based | 95 | THETA | 0.858 | 0.747 | 0.948 | 8.426 |  |

### Fallback counts

| model | fb_point | fb_cal | fb_mb | n_forecast_rows |
|---|---|---|---|---|
| NAIVE | 0 | 0 | 0 | 960 |
| SNAIVE | 0 | 0 | 0 | 960 |
| DRIFT | 0 | 0 | 0 | 960 |
| SMEAN | 0 | 0 | 0 | 960 |
| ETS | 0 | 0 | 0 | 960 |
| ARIMA | 0 | 0 | 0 | 960 |
| THETA | 0 | 0 | 0 | 960 |
| ML-G | 0 | 0 | 0 | 960 |
| ML-L | 0 | 0 | 0 | 960 |
| COMB | 0 | 0 | 0 | 960 |

### Claims

| Claim | Value | Criterion met | CI95 | p+ | p- |
|---|---|---|---|---|---|
| H1: ML-G beats SNAIVE by >= 0.03 MASE | 0.333 | False | [0.167, 0.500] | 0.9993 | 0.0029 |
| H2: STAT* beats SNAIVE | 0.467 | False | [0.300, 0.633] | 0.9509 | 0.1009 |
| H3a: ETS conformal 80% | 0.877 | False | [0.845, 0.906] | 0.9974 | 0.0026 |
| H3a: ARIMA conformal 80% | 0.880 | False | [0.849, 0.908] | 0.9983 | 0.0017 |
| H3a: ML-G conformal 80% | 0.856 | False | [0.818, 0.892] | 0.9107 | 0.0893 |
| H3a: ETS conformal 95% | 0.968 | True | [0.944, 0.988] | 0.1348 | 1.0000 |
| H3a: ARIMA conformal 95% | 0.968 | True | [0.944, 0.988] | 0.1460 | 1.0000 |
| H3a: ML-G conformal 95% | 0.963 | True | [0.936, 0.983] | 0.0625 | 1.0000 |
| H3b: ETS model-based 80% | 0.796 | False | [0.725, 0.856] | 0.7713 | 0.2287 |
| H3b: ARIMA model-based 80% | 0.758 | True | [0.695, 0.814] | 0.3495 | 0.6505 |
| H3b: ETS model-based 95% | 0.908 | True | [0.865, 0.945] | 0.2859 | 0.7141 |
| H3b: ARIMA model-based 95% | 0.871 | True | [0.820, 0.916] | 0.0151 | 0.9849 |
| H4 | 1.000 | True |  | 0.3819 | 0.6181 |

## daily

### Error by horizon

| model | h | MASE_mean | MASE_median | sMAPE_mean | RMSSE_mean |
|---|---|---|---|---|---|
| NAIVE | 1 | 0.603 | 0.542 | 4.700 | 0.538 |
| NAIVE | 7 | 0.925 | 0.837 | 7.403 | 0.825 |
| NAIVE | 14 | 1.043 | 0.997 | 8.324 | 0.880 |
| NAIVE | 28 | 1.623 | 1.570 | 12.067 | 1.378 |
| SNAIVE | 1 | 1.192 | 1.165 | 8.839 | 1.038 |
| SNAIVE | 7 | 0.925 | 0.837 | 7.403 | 0.825 |
| SNAIVE | 14 | 1.043 | 0.997 | 8.324 | 0.880 |
| SNAIVE | 28 | 1.623 | 1.570 | 12.067 | 1.378 |
| DRIFT | 1 | 0.604 | 0.542 | 4.707 | 0.539 |
| DRIFT | 7 | 0.937 | 0.859 | 7.473 | 0.835 |
| DRIFT | 14 | 1.055 | 1.011 | 8.420 | 0.890 |
| DRIFT | 28 | 1.668 | 1.593 | 12.347 | 1.408 |
| SMEAN | 1 | 1.239 | 1.214 | 10.079 | 1.017 |
| SMEAN | 7 | 1.038 | 0.919 | 8.814 | 0.901 |
| SMEAN | 14 | 1.212 | 1.118 | 10.114 | 1.026 |
| SMEAN | 28 | 1.806 | 1.811 | 13.788 | 1.504 |
| ETS | 1 | 0.699 | 0.588 | 5.737 | 0.601 |
| ETS | 7 | 0.940 | 0.884 | 7.645 | 0.847 |
| ETS | 14 | 1.055 | 1.043 | 8.538 | 0.891 |
| ETS | 28 | 1.587 | 1.425 | 12.251 | 1.327 |
| ARIMA | 1 | 0.613 | 0.544 | 5.000 | 0.522 |
| ARIMA | 7 | 0.938 | 0.884 | 8.052 | 0.812 |
| ARIMA | 14 | 1.061 | 1.036 | 9.052 | 0.884 |
| ARIMA | 28 | 1.639 | 1.480 | 12.656 | 1.363 |
| THETA | 1 | 0.625 | 0.523 | 4.982 | 0.551 |
| THETA | 7 | 0.897 | 0.802 | 7.249 | 0.810 |
| THETA | 14 | 1.013 | 1.023 | 7.998 | 0.856 |
| THETA | 28 | 1.560 | 1.466 | 11.654 | 1.313 |
| ML-G | 1 | 0.635 | 0.564 | 4.968 | 0.560 |
| ML-G | 7 | 1.053 | 0.955 | 8.305 | 0.968 |
| ML-G | 14 | 1.188 | 1.111 | 9.302 | 1.039 |
| ML-G | 28 | 1.474 | 1.404 | 11.436 | 1.252 |
| ML-L | 1 | 0.647 | 0.591 | 5.166 | 0.551 |
| ML-L | 7 | 1.044 | 0.912 | 8.477 | 0.917 |
| ML-L | 14 | 1.208 | 1.184 | 9.492 | 1.016 |
| ML-L | 28 | 1.341 | 1.397 | 10.090 | 1.124 |
| COMB | 1 | 0.622 | 0.527 | 4.972 | 0.537 |
| COMB | 7 | 0.932 | 0.896 | 7.628 | 0.842 |
| COMB | 14 | 1.027 | 0.964 | 8.359 | 0.878 |
| COMB | 28 | 1.451 | 1.336 | 11.409 | 1.217 |

### Win rates against SNAIVE

|  | win_rate_margin0 | win_rate_margin0.03 | mean_MASE_ref_h | wilcoxon_p_secondary |
|---|---|---|---|---|
| ETS | 0.633 | 0.600 | 1.587 | 0.080 |
| ARIMA | 0.500 | 0.433 | 1.639 | 0.871 |
| THETA | 0.667 | 0.600 | 1.560 | 0.002 |
| ML-G | 0.633 | 0.600 | 1.474 | 0.129 |
| ML-L | 0.700 | 0.700 | 1.341 | 0.007 |
| COMB | 0.733 | 0.700 | 1.451 | 0.001 |
| STAT* | 0.633 | 0.567 | 1.604 | 0.428 |

### Intervals

| interval | level | model | coverage | ci95_lo | ci95_hi | mean_scaled_width | mean_pinball80 |
|---|---|---|---|---|---|---|---|
| conformal | 80 | NAIVE | 0.804 | 0.777 | 0.829 | 3.624 | 0.368 |
| conformal | 80 | SNAIVE | 0.768 | 0.741 | 0.794 | 3.791 | 0.414 |
| conformal | 80 | DRIFT | 0.800 | 0.774 | 0.824 | 3.645 | 0.374 |
| conformal | 80 | SMEAN | 0.727 | 0.702 | 0.751 | 3.728 | 0.443 |
| conformal | 80 | ETS | 0.790 | 0.765 | 0.815 | 3.355 | 0.366 |
| conformal | 80 | ARIMA | 0.787 | 0.758 | 0.816 | 3.296 | 0.357 |
| conformal | 80 | THETA | 0.782 | 0.754 | 0.811 | 3.307 | 0.356 |
| conformal | 80 | ML-G | 0.780 | 0.753 | 0.808 | 3.257 | 0.375 |
| conformal | 80 | ML-L | 0.783 | 0.759 | 0.806 | 3.293 | 0.355 |
| conformal | 80 | COMB | 0.782 | 0.759 | 0.805 | 3.090 | 0.344 |
| conformal | 95 | NAIVE | 0.929 | 0.909 | 0.947 | 6.293 |  |
| conformal | 95 | SNAIVE | 0.918 | 0.900 | 0.934 | 6.772 |  |
| conformal | 95 | DRIFT | 0.927 | 0.908 | 0.944 | 6.323 |  |
| conformal | 95 | SMEAN | 0.896 | 0.877 | 0.914 | 6.373 |  |
| conformal | 95 | ETS | 0.941 | 0.926 | 0.954 | 5.877 |  |
| conformal | 95 | ARIMA | 0.934 | 0.918 | 0.950 | 5.711 |  |
| conformal | 95 | THETA | 0.922 | 0.903 | 0.939 | 5.771 |  |
| conformal | 95 | ML-G | 0.921 | 0.904 | 0.938 | 5.920 |  |
| conformal | 95 | ML-L | 0.930 | 0.915 | 0.944 | 5.690 |  |
| conformal | 95 | COMB | 0.933 | 0.918 | 0.948 | 5.451 |  |
| model-based | 80 | ETS | 0.916 | 0.896 | 0.934 | 4.802 | 0.366 |
| model-based | 80 | ARIMA | 0.894 | 0.872 | 0.916 | 4.326 | 0.356 |
| model-based | 80 | THETA | 0.964 | 0.941 | 0.980 | 10.732 | 0.543 |
| model-based | 95 | ETS | 0.979 | 0.964 | 0.992 | 7.367 |  |
| model-based | 95 | ARIMA | 0.973 | 0.957 | 0.985 | 6.615 |  |
| model-based | 95 | THETA | 0.988 | 0.972 | 0.998 | 16.413 |  |

### Fallback counts

| model | fb_point | fb_cal | fb_mb | n_forecast_rows |
|---|---|---|---|---|
| NAIVE | 0 | 0 | 0 | 960 |
| SNAIVE | 0 | 0 | 0 | 960 |
| DRIFT | 0 | 0 | 0 | 960 |
| SMEAN | 0 | 0 | 0 | 960 |
| ETS | 0 | 0 | 0 | 960 |
| ARIMA | 0 | 0 | 0 | 960 |
| THETA | 0 | 0 | 0 | 960 |
| ML-G | 0 | 0 | 0 | 960 |
| ML-L | 0 | 0 | 0 | 960 |
| COMB | 0 | 0 | 0 | 960 |

### Claims

| Claim | Value | Criterion met | CI95 | p+ | p- |
|---|---|---|---|---|---|
| H1: ML-G beats SNAIVE by >= 0.03 MASE | 0.600 | True | [0.433, 0.767] | 0.5745 | 0.5718 |
| H2: STAT* beats SNAIVE | 0.633 | True | [0.467, 0.800] | 0.4182 | 0.7148 |
| H3a: ETS conformal 80% | 0.790 | True | [0.765, 0.815] | 0.0651 | 1.0000 |
| H3a: ARIMA conformal 80% | 0.787 | True | [0.758, 0.816] | 0.1269 | 1.0000 |
| H3a: ML-G conformal 80% | 0.780 | True | [0.753, 0.808] | 0.2453 | 1.0000 |
| H3a: ETS conformal 95% | 0.941 | True | [0.926, 0.954] | 0.0028 | 1.0000 |
| H3a: ARIMA conformal 95% | 0.934 | True | [0.918, 0.950] | 0.0481 | 1.0000 |
| H3a: ML-G conformal 95% | 0.921 | True | [0.904, 0.938] | 0.4806 | 1.0000 |
| H3b: ETS model-based 80% | 0.916 | False | [0.896, 0.934] | 1.0000 | 0.0000 |
| H3b: ARIMA model-based 80% | 0.894 | False | [0.872, 0.916] | 1.0000 | 0.0000 |
| H3b: ETS model-based 95% | 0.979 | False | [0.964, 0.992] | 1.0000 | 0.0000 |
| H3b: ARIMA model-based 95% | 0.973 | False | [0.957, 0.985] | 1.0000 | 0.0000 |
