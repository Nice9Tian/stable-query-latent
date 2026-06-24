# VICReg PXI Result

Generated: 2026-06-24 19:25:14

## Scope

- Evaluation set: intersection of `VICReg_review/tags/game_descriptions/*.txt` and `PXIbench_test/pxi_vicreg_overlap.json` / PXIbenchmark games.
- Matched games with real description text: 21 / 21 PXI-VICReg overlap games.
- Text source: `VICReg_review/tags/game_descriptions/{appid}.txt` (Steam/appdetails-derived game descriptions, mostly Chinese).
- Prediction path: game description text -> Qwen sentence embeddings -> frozen VICReg encoder -> PXI mean-regression head.
- Encoder: `VICReg_review/heads/sweep_adv/vicreg_adv10_best.pt` resolved from tag probe `VICReg_review/heads/tag_probe_linear.pt`.
- PXI head: `VICReg_review/heads/pxi_probe_linear.pt` (`target_kind=mean_regression`, trained on 21 PXI/VICReg overlap games).
- Feature view policy: `4` deterministic sub-sampled views, sample_fraction=0.6, matching the current Validation path.
- Clipping: 199/210 predicted dimension values were clipped to the head target range.

## Overall Metrics

| subset | N values | MAE | RMSE | Pearson | R^2 |
|---|---:|---:|---:|---:|---:|
| all dimensions | 210 | 1.607 | 1.961 | 0.205 | -3.635 |
| functional | 105 | 1.418 | 1.826 | 0.318 | -2.679 |
| psychological | 105 | 1.796 | 2.088 | -0.064 | -4.806 |

## Per-Game Summary

| appid | PXI game | Steam game | match | PXI samples | sentences | chars | MAE | RMSE | functional MAE | psychological MAE |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1407200 | World of Tanks | World of Tanks | exact | 5 | 27 | 2357 | 0.628 | 0.914 | 0.734 | 0.521 |
| 1549970 | Aliens Fireteam Elite | Aliens: Fireteam Elite | exact | 1 | 36 | 2131 | 0.778 | 1.004 | 0.984 | 0.572 |
| 1237950 | Star Wars Battlefront | STAR WARS™ Battlefront™ II | vetted_variant | 1 | 28 | 2300 | 0.932 | 1.118 | 1.035 | 0.828 |
| 1343400 | RuneScape | RuneScape ® | exact | 4 | 28 | 1931 | 1.337 | 1.644 | 1.534 | 1.140 |
| 1593500 | God of War | God of War | exact | 3 | 27 | 1156 | 1.413 | 1.759 | 1.087 | 1.739 |
| 1172470 | Apex Legends | Apex Legends™ | exact | 3 | 29 | 1428 | 1.432 | 1.961 | 1.379 | 1.485 |
| 1293830 | Forza Horizon 4 | Forza Horizon 4 | exact | 1 | 25 | 1472 | 1.453 | 1.802 | 1.732 | 1.174 |
| 1124300 | Humankind | HUMANKIND™ | exact | 1 | 35 | 1897 | 1.486 | 1.964 | 0.665 | 2.308 |
| 1817070 | Spider-Man | Marvel’s Spider-Man Remastered | vetted_variant | 2 | 41 | 2153 | 1.487 | 1.844 | 1.200 | 1.774 |
| 1113560 | Nier Replicant | NieR Replicant™ ver.1.22474487139... | vetted_variant | 1 | 53 | 1238 | 1.521 | 1.656 | 1.400 | 1.642 |
| 1659420 | Uncharted | UNCHARTED™: Legacy of Thieves Collection | vetted_variant | 1 | 28 | 3160 | 1.588 | 1.735 | 1.534 | 1.642 |
| 1284210 | Guild Wars 2 | Guild Wars 2 | exact | 13 | 28 | 2374 | 1.618 | 1.889 | 1.493 | 1.743 |
| 1091500 | Cyberpunk 2077 | Cyberpunk 2077 | exact | 8 | 17 | 675 | 1.712 | 1.877 | 1.609 | 1.816 |
| 1145360 | Hades | Hades | exact | 13 | 26 | 2698 | 1.811 | 2.041 | 1.539 | 2.082 |
| 1262540 | Need for Speed | Need for Speed™ | exact | 4 | 22 | 1091 | 1.816 | 2.161 | 1.274 | 2.357 |
| 1649240 | Returnal | Returnal™ | exact | 1 | 28 | 2493 | 1.955 | 2.422 | 1.068 | 2.842 |
| 1458100 | Cozy Grove | Cozy Grove | exact | 1 | 22 | 1173 | 2.020 | 2.507 | 1.534 | 2.506 |
| 1449560 | Metro Exodus | Metro Exodus | exact | 1 | 27 | 1228 | 2.046 | 2.215 | 1.918 | 2.174 |
| 1151640 | Horizon zero dawn | Horizon Zero Dawn™ Complete Edition | vetted_variant | 3 | 33 | 1180 | 2.143 | 2.245 | 2.134 | 2.151 |
| 1325200 | Nioh 2 | Nioh 2 – The Complete Edition | vetted_variant | 1 | 46 | 1628 | 2.254 | 2.584 | 2.134 | 2.374 |
| 1237970 | Titanfall 2 | Titanfall® 2 | exact | 1 | 19 | 1293 | 2.314 | 2.659 | 1.788 | 2.840 |

## Per-Dimension Metrics

| dimension | group | MAE | RMSE | Pearson | R^2 | actual mean | predicted mean | raw min | raw max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| psychological_meaning | psychological | 1.675 | 1.879 | -0.228 | -3.982 | 1.749 | 0.074 | -19.176 | 0.242 |
| psychological_mastery | psychological | 1.179 | 1.339 | nan | -3.446 | 1.913 | 0.734 | -19.128 | -1.987 |
| psychological_curiosity | psychological | 1.920 | 2.186 | nan | -3.372 | 1.920 | 0.000 | -21.623 | -3.912 |
| psychological_autonomy | psychological | 2.878 | 3.059 | -0.144 | -9.331 | 1.944 | -0.934 | -36.994 | 0.388 |
| psychological_immersion | psychological | 1.326 | 1.529 | nan | -3.037 | 1.656 | 0.330 | -12.296 | -0.455 |
| functional_progress_feedback | functional | 1.225 | 1.440 | nan | -2.613 | 1.775 | 3.000 | 4.509 | 17.220 |
| functional_ease_of_control | functional | 0.789 | 0.973 | nan | -1.924 | 1.881 | 2.670 | 5.758 | 14.910 |
| functional_audiovisual_appeal | functional | 1.434 | 1.559 | 0.116 | -6.195 | 2.546 | 1.112 | -5.661 | 2.007 |
| functional_goals_and_rules | functional | 0.728 | 1.007 | nan | -1.095 | 2.272 | 3.000 | 4.159 | 38.197 |
| functional_challenge | functional | 2.913 | 3.193 | 0.047 | -5.414 | 1.094 | -1.819 | -25.521 | -0.538 |

## Predictions

### Aliens Fireteam Elite (1549970)

- Steam name: Aliens: Fireteam Elite; match: `exact`; genre: Third person shooter; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1549970.txt`; sentences: 36; chars: 2131.
- MAE: 0.778; RMSE: 1.004; functional MAE: 0.984; psychological MAE: 0.572.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 1.330 | 0.066 | -1.264 | -0.314 |
| psychological_mastery | 1.330 | 0.734 | -0.596 | -2.660 |
| psychological_curiosity | 1.000 | 0.000 | -1.000 | -4.731 |
| psychological_autonomy | -1.000 | -1.000 | +0.000 | -4.326 |
| psychological_immersion | 0.330 | 0.330 | +0.000 | -2.367 |
| functional_progress_feedback | 2.000 | 3.000 | +1.000 | 7.188 |
| functional_ease_of_control | 1.330 | 2.670 | +1.340 | 6.080 |
| functional_audiovisual_appeal | 1.670 | 1.092 | -0.578 | 1.092 |
| functional_goals_and_rules | 3.000 | 3.000 | +0.000 | 10.499 |
| functional_challenge | 0.000 | -2.000 | -2.000 | -2.844 |

### Apex Legends (1172470)

- Steam name: Apex Legends™; match: `exact`; genre: First person shooter; PXI samples: 3.
- Text: `VICReg_review/tags/game_descriptions/1172470.txt`; sentences: 29; chars: 1428.
- MAE: 1.432; RMSE: 1.961; functional MAE: 1.379; psychological MAE: 1.485.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 0.333 | 0.066 | -0.267 | -5.986 |
| psychological_mastery | 1.443 | 0.734 | -0.709 | -6.429 |
| psychological_curiosity | 0.887 | 0.000 | -0.887 | -11.145 |
| psychological_autonomy | 2.667 | -1.000 | -3.667 | -5.344 |
| psychological_immersion | 2.223 | 0.330 | -1.893 | -3.219 |
| functional_progress_feedback | 2.223 | 3.000 | +0.777 | 8.036 |
| functional_ease_of_control | 2.333 | 2.670 | +0.337 | 9.019 |
| functional_audiovisual_appeal | 2.557 | 1.000 | -1.557 | -1.640 |
| functional_goals_and_rules | 2.890 | 3.000 | +0.110 | 10.867 |
| functional_challenge | 2.113 | -2.000 | -4.113 | -3.547 |

### Cozy Grove (1458100)

- Steam name: Cozy Grove; match: `exact`; genre: Action adventure game; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1458100.txt`; sentences: 22; chars: 1173.
- MAE: 2.020; RMSE: 2.507; functional MAE: 1.534; psychological MAE: 2.506.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 2.670 | 0.066 | -2.604 | -12.081 |
| psychological_mastery | 2.330 | 0.734 | -1.596 | -14.454 |
| psychological_curiosity | 3.000 | 0.000 | -3.000 | -13.357 |
| psychological_autonomy | 2.330 | -1.000 | -3.330 | -27.843 |
| psychological_immersion | 2.330 | 0.330 | -2.000 | -10.003 |
| functional_progress_feedback | 3.000 | 3.000 | +0.000 | 14.795 |
| functional_ease_of_control | 2.000 | 2.670 | +0.670 | 10.728 |
| functional_audiovisual_appeal | 3.000 | 1.000 | -2.000 | -3.811 |
| functional_goals_and_rules | 3.000 | 3.000 | +0.000 | 30.756 |
| functional_challenge | 3.000 | -2.000 | -5.000 | -18.607 |

### Cyberpunk 2077 (1091500)

- Steam name: Cyberpunk 2077; match: `exact`; genre: Action role playing game; PXI samples: 8.
- Text: `VICReg_review/tags/game_descriptions/1091500.txt`; sentences: 17; chars: 675.
- MAE: 1.712; RMSE: 1.877; functional MAE: 1.609; psychological MAE: 1.816.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 1.042 | 0.066 | -0.977 | -7.589 |
| psychological_mastery | 1.834 | 0.734 | -1.100 | -10.732 |
| psychological_curiosity | 2.375 | 0.000 | -2.375 | -9.930 |
| psychological_autonomy | 2.291 | -1.000 | -3.291 | -21.560 |
| psychological_immersion | 1.666 | 0.330 | -1.336 | -7.095 |
| functional_progress_feedback | 1.666 | 3.000 | +1.334 | 12.779 |
| functional_ease_of_control | 1.417 | 2.670 | +1.253 | 9.317 |
| functional_audiovisual_appeal | 2.334 | 1.000 | -1.334 | -1.540 |
| functional_goals_and_rules | 1.709 | 3.000 | +1.291 | 25.384 |
| functional_challenge | 0.834 | -2.000 | -2.834 | -14.972 |

### Forza Horizon 4 (1293830)

- Steam name: Forza Horizon 4; match: `exact`; genre: Racing game; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1293830.txt`; sentences: 25; chars: 1472.
- MAE: 1.453; RMSE: 1.802; functional MAE: 1.732; psychological MAE: 1.174.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 1.000 | 0.066 | -0.934 | -19.176 |
| psychological_mastery | 1.000 | 0.734 | -0.266 | -19.128 |
| psychological_curiosity | 0.000 | 0.000 | +0.000 | -21.623 |
| psychological_autonomy | 2.670 | -1.000 | -3.670 | -25.604 |
| psychological_immersion | 1.330 | 0.330 | -1.000 | -6.393 |
| functional_progress_feedback | 0.670 | 3.000 | +2.330 | 13.878 |
| functional_ease_of_control | 2.000 | 2.670 | +0.670 | 14.910 |
| functional_audiovisual_appeal | 2.330 | 1.000 | -1.330 | -5.661 |
| functional_goals_and_rules | 0.670 | 3.000 | +2.330 | 27.181 |
| functional_challenge | 0.000 | -2.000 | -2.000 | -14.061 |

### God of War (1593500)

- Steam name: God of War; match: `exact`; genre: Action adventure game; PXI samples: 3.
- Text: `VICReg_review/tags/game_descriptions/1593500.txt`; sentences: 27; chars: 1156.
- MAE: 1.413; RMSE: 1.759; functional MAE: 1.087; psychological MAE: 1.739.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 0.890 | 0.242 | -0.648 | 0.242 |
| psychological_mastery | 2.110 | 0.734 | -1.376 | -1.987 |
| psychological_curiosity | 2.780 | 0.000 | -2.780 | -4.505 |
| psychological_autonomy | 1.333 | -1.000 | -2.333 | -1.814 |
| psychological_immersion | 1.890 | 0.330 | -1.560 | -0.455 |
| functional_progress_feedback | 2.223 | 3.000 | +0.777 | 6.032 |
| functional_ease_of_control | 2.333 | 2.670 | +0.337 | 5.841 |
| functional_audiovisual_appeal | 2.333 | 1.799 | -0.534 | 1.799 |
| functional_goals_and_rules | 2.667 | 3.000 | +0.333 | 7.900 |
| functional_challenge | 2.223 | -1.229 | -3.453 | -1.229 |

### Guild Wars 2 (1284210)

- Steam name: Guild Wars 2; match: `exact`; genre: MMO role playing game; PXI samples: 13.
- Text: `VICReg_review/tags/game_descriptions/1284210.txt`; sentences: 28; chars: 2374.
- MAE: 1.618; RMSE: 1.889; functional MAE: 1.493; psychological MAE: 1.743.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 1.974 | 0.066 | -1.908 | -2.045 |
| psychological_mastery | 2.051 | 0.734 | -1.317 | -3.062 |
| psychological_curiosity | 1.359 | 0.000 | -1.359 | -6.078 |
| psychological_autonomy | 2.435 | -1.000 | -3.435 | -3.142 |
| psychological_immersion | 1.026 | 0.330 | -0.696 | -2.612 |
| functional_progress_feedback | 1.667 | 3.000 | +1.333 | 6.662 |
| functional_ease_of_control | 2.025 | 2.670 | +0.645 | 6.570 |
| functional_audiovisual_appeal | 2.231 | 1.000 | -1.231 | -0.031 |
| functional_goals_and_rules | 2.179 | 3.000 | +0.821 | 8.913 |
| functional_challenge | 1.435 | -2.000 | -3.435 | -2.099 |

### Hades (1145360)

- Steam name: Hades; match: `exact`; genre: Action role playing game; PXI samples: 13.
- Text: `VICReg_review/tags/game_descriptions/1145360.txt`; sentences: 26; chars: 2698.
- MAE: 1.811; RMSE: 2.041; functional MAE: 1.539; psychological MAE: 2.082.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 1.872 | 0.066 | -1.806 | -8.617 |
| psychological_mastery | 2.051 | 0.734 | -1.317 | -12.240 |
| psychological_curiosity | 2.770 | 0.000 | -2.770 | -15.244 |
| psychological_autonomy | 1.744 | -1.000 | -2.744 | -15.865 |
| psychological_immersion | 2.103 | 0.330 | -1.773 | -2.970 |
| functional_progress_feedback | 1.922 | 3.000 | +1.078 | 12.121 |
| functional_ease_of_control | 1.974 | 2.670 | +0.696 | 12.260 |
| functional_audiovisual_appeal | 2.898 | 1.000 | -1.898 | -0.820 |
| functional_goals_and_rules | 2.565 | 3.000 | +0.435 | 20.648 |
| functional_challenge | 1.589 | -2.000 | -3.589 | -8.858 |

### Horizon zero dawn (1151640)

- Steam name: Horizon Zero Dawn™ Complete Edition; match: `vetted_variant`; genre: Action role playing game; PXI samples: 3.
- Text: `VICReg_review/tags/game_descriptions/1151640.txt`; sentences: 33; chars: 1180.
- MAE: 2.143; RMSE: 2.245; functional MAE: 2.134; psychological MAE: 2.151.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 2.223 | 0.066 | -2.157 | -3.772 |
| psychological_mastery | 2.443 | 0.734 | -1.709 | -8.045 |
| psychological_curiosity | 2.667 | 0.000 | -2.667 | -6.928 |
| psychological_autonomy | 2.110 | -1.000 | -3.110 | -18.528 |
| psychological_immersion | 1.443 | 0.330 | -1.113 | -4.279 |
| functional_progress_feedback | 0.887 | 3.000 | +2.113 | 11.450 |
| functional_ease_of_control | 0.777 | 2.670 | +1.893 | 8.005 |
| functional_audiovisual_appeal | 2.443 | 1.000 | -1.443 | 0.745 |
| functional_goals_and_rules | 1.113 | 3.000 | +1.887 | 23.174 |
| functional_challenge | 1.333 | -2.000 | -3.333 | -11.660 |

### Humankind (1124300)

- Steam name: HUMANKIND™; match: `exact`; genre: Strategy game; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1124300.txt`; sentences: 35; chars: 1897.
- MAE: 1.486; RMSE: 1.964; functional MAE: 0.665; psychological MAE: 2.308.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 3.000 | 0.066 | -2.934 | -1.366 |
| psychological_mastery | 2.670 | 0.734 | -1.936 | -5.826 |
| psychological_curiosity | 3.000 | 0.000 | -3.000 | -7.305 |
| psychological_autonomy | 2.670 | -1.000 | -3.670 | -10.731 |
| psychological_immersion | 0.330 | 0.330 | +0.000 | -1.182 |
| functional_progress_feedback | 1.670 | 3.000 | +1.330 | 9.624 |
| functional_ease_of_control | 1.670 | 2.670 | +1.000 | 8.193 |
| functional_audiovisual_appeal | 3.000 | 2.007 | -0.993 | 2.007 |
| functional_goals_and_rules | 3.000 | 3.000 | +0.000 | 16.875 |
| functional_challenge | -2.000 | -2.000 | +0.000 | -5.132 |

### Metro Exodus (1449560)

- Steam name: Metro Exodus; match: `exact`; genre: First person shooter; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1449560.txt`; sentences: 27; chars: 1228.
- MAE: 2.046; RMSE: 2.215; functional MAE: 1.918; psychological MAE: 2.174.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 2.330 | 0.066 | -2.264 | -1.389 |
| psychological_mastery | 1.670 | 0.734 | -0.936 | -4.704 |
| psychological_curiosity | 3.000 | 0.000 | -3.000 | -4.899 |
| psychological_autonomy | 2.000 | -1.000 | -3.000 | -10.629 |
| psychological_immersion | 2.000 | 0.330 | -1.670 | -3.435 |
| functional_progress_feedback | 1.000 | 3.000 | +2.000 | 9.219 |
| functional_ease_of_control | 0.330 | 2.670 | +2.340 | 6.824 |
| functional_audiovisual_appeal | 3.000 | 1.079 | -1.921 | 1.079 |
| functional_goals_and_rules | 2.670 | 3.000 | +0.330 | 16.226 |
| functional_challenge | 1.000 | -2.000 | -3.000 | -6.692 |

### Need for Speed (1262540)

- Steam name: Need for Speed™; match: `exact`; genre: Racing game; PXI samples: 4.
- Text: `VICReg_review/tags/game_descriptions/1262540.txt`; sentences: 22; chars: 1091.
- MAE: 1.816; RMSE: 2.161; functional MAE: 1.274; psychological MAE: 2.357.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 1.583 | 0.066 | -1.517 | -0.399 |
| psychological_mastery | 2.335 | 0.734 | -1.601 | -3.237 |
| psychological_curiosity | 2.917 | 0.000 | -2.917 | -3.912 |
| psychological_autonomy | 2.665 | -1.000 | -3.665 | -8.351 |
| psychological_immersion | 2.417 | 0.330 | -2.087 | -2.573 |
| functional_progress_feedback | 2.750 | 3.000 | +0.250 | 7.979 |
| functional_ease_of_control | 2.500 | 2.670 | +0.170 | 5.758 |
| functional_audiovisual_appeal | 2.917 | 1.385 | -1.533 | 1.385 |
| functional_goals_and_rules | 2.083 | 3.000 | +0.917 | 14.160 |
| functional_challenge | 1.500 | -2.000 | -3.500 | -5.350 |

### Nier Replicant (1113560)

- Steam name: NieR Replicant™ ver.1.22474487139...; match: `vetted_variant`; genre: Action role playing game; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1113560.txt`; sentences: 53; chars: 1238.
- MAE: 1.521; RMSE: 1.656; functional MAE: 1.400; psychological MAE: 1.642.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 2.000 | 0.066 | -1.934 | -14.405 |
| psychological_mastery | 1.670 | 0.734 | -0.936 | -17.980 |
| psychological_curiosity | 2.670 | 0.000 | -2.670 | -14.529 |
| psychological_autonomy | 1.000 | -1.000 | -2.000 | -36.994 |
| psychological_immersion | 1.000 | 0.330 | -0.670 | -12.296 |
| functional_progress_feedback | 1.670 | 3.000 | +1.330 | 17.220 |
| functional_ease_of_control | 2.000 | 2.670 | +0.670 | 11.482 |
| functional_audiovisual_appeal | 3.000 | 1.000 | -2.000 | -4.376 |
| functional_goals_and_rules | 2.000 | 3.000 | +1.000 | 38.197 |
| functional_challenge | 0.000 | -2.000 | -2.000 | -25.521 |

### Nioh 2 (1325200)

- Steam name: Nioh 2 – The Complete Edition; match: `vetted_variant`; genre: Action role playing game; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1325200.txt`; sentences: 46; chars: 1628.
- MAE: 2.254; RMSE: 2.584; functional MAE: 2.134; psychological MAE: 2.374.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 2.330 | 0.066 | -2.264 | -2.906 |
| psychological_mastery | 2.670 | 0.734 | -1.936 | -5.920 |
| psychological_curiosity | 2.670 | 0.000 | -2.670 | -5.421 |
| psychological_autonomy | 3.000 | -1.000 | -4.000 | -12.509 |
| psychological_immersion | 1.330 | 0.330 | -1.000 | -4.780 |
| functional_progress_feedback | 2.000 | 3.000 | +1.000 | 9.881 |
| functional_ease_of_control | 1.670 | 2.670 | +1.000 | 7.096 |
| functional_audiovisual_appeal | 3.000 | 1.000 | -2.000 | 0.323 |
| functional_goals_and_rules | 1.330 | 3.000 | +1.670 | 17.544 |
| functional_challenge | 3.000 | -2.000 | -5.000 | -8.764 |

### Returnal (1649240)

- Steam name: Returnal™; match: `exact`; genre: Third person shooter; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1649240.txt`; sentences: 28; chars: 2493.
- MAE: 1.955; RMSE: 2.422; functional MAE: 1.068; psychological MAE: 2.842.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 2.670 | 0.066 | -2.604 | -2.266 |
| psychological_mastery | 3.000 | 0.734 | -2.266 | -3.705 |
| psychological_curiosity | 3.000 | 0.000 | -3.000 | -4.683 |
| psychological_autonomy | 3.000 | -1.000 | -4.000 | -5.828 |
| psychological_immersion | 2.670 | 0.330 | -2.340 | -3.464 |
| functional_progress_feedback | 3.000 | 3.000 | +0.000 | 7.334 |
| functional_ease_of_control | 2.670 | 2.670 | +0.000 | 6.031 |
| functional_audiovisual_appeal | 2.670 | 1.000 | -1.670 | 0.207 |
| functional_goals_and_rules | 3.000 | 3.000 | +0.000 | 11.027 |
| functional_challenge | 1.670 | -2.000 | -3.670 | -4.878 |

### RuneScape (1343400)

- Steam name: RuneScape ®; match: `exact`; genre: MMO role playing game; PXI samples: 4.
- Text: `VICReg_review/tags/game_descriptions/1343400.txt`; sentences: 28; chars: 1931.
- MAE: 1.337; RMSE: 1.644; functional MAE: 1.534; psychological MAE: 1.140.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 0.417 | 0.066 | -0.352 | -2.041 |
| psychological_mastery | 1.500 | 0.734 | -0.766 | -3.544 |
| psychological_curiosity | 1.000 | 0.000 | -1.000 | -5.784 |
| psychological_autonomy | 2.250 | -1.000 | -3.250 | -4.279 |
| psychological_immersion | 0.665 | 0.330 | -0.335 | -2.371 |
| functional_progress_feedback | 1.165 | 3.000 | +1.835 | 6.757 |
| functional_ease_of_control | 1.335 | 2.670 | +1.335 | 6.360 |
| functional_audiovisual_appeal | 1.250 | 1.000 | -0.250 | 0.399 |
| functional_goals_and_rules | 1.335 | 3.000 | +1.665 | 9.587 |
| functional_challenge | 0.583 | -2.000 | -2.583 | -3.490 |

### Spider-Man (1817070)

- Steam name: Marvel’s Spider-Man Remastered; match: `vetted_variant`; genre: Action adventure game; PXI samples: 2.
- Text: `VICReg_review/tags/game_descriptions/1817070.txt`; sentences: 41; chars: 2153.
- MAE: 1.487; RMSE: 1.844; functional MAE: 1.200; psychological MAE: 1.774.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 2.335 | 0.066 | -2.269 | -6.237 |
| psychological_mastery | 1.335 | 0.734 | -0.601 | -10.184 |
| psychological_curiosity | 1.500 | 0.000 | -1.500 | -11.496 |
| psychological_autonomy | 1.665 | -1.000 | -2.665 | -17.600 |
| psychological_immersion | 2.165 | 0.330 | -1.835 | -5.051 |
| functional_progress_feedback | 2.835 | 3.000 | +0.165 | 12.084 |
| functional_ease_of_control | 2.335 | 2.670 | +0.335 | 10.069 |
| functional_audiovisual_appeal | 2.835 | 1.000 | -1.835 | -0.632 |
| functional_goals_and_rules | 2.835 | 3.000 | +0.165 | 22.256 |
| functional_challenge | 1.500 | -2.000 | -3.500 | -11.607 |

### Star Wars Battlefront (1237950)

- Steam name: STAR WARS™ Battlefront™ II; match: `vetted_variant`; genre: First person shooter; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1237950.txt`; sentences: 28; chars: 2300.
- MAE: 0.932; RMSE: 1.118; functional MAE: 1.035; psychological MAE: 0.828.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 2.000 | 0.066 | -1.934 | -4.618 |
| psychological_mastery | 1.000 | 0.734 | -0.266 | -2.899 |
| psychological_curiosity | 0.000 | 0.000 | +0.000 | -7.333 |
| psychological_autonomy | 1.330 | 0.388 | -0.942 | 0.388 |
| psychological_immersion | 1.330 | 0.330 | -1.000 | -3.035 |
| functional_progress_feedback | 2.000 | 3.000 | +1.000 | 4.509 |
| functional_ease_of_control | 2.330 | 2.670 | +0.340 | 6.104 |
| functional_audiovisual_appeal | 3.000 | 1.000 | -2.000 | -1.921 |
| functional_goals_and_rules | 2.000 | 3.000 | +1.000 | 4.159 |
| functional_challenge | 0.000 | -0.835 | -0.835 | -0.835 |

### Titanfall 2 (1237970)

- Steam name: Titanfall® 2; match: `exact`; genre: First person shooter; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1237970.txt`; sentences: 19; chars: 1293.
- MAE: 2.314; RMSE: 2.659; functional MAE: 1.788; psychological MAE: 2.840.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 3.000 | 0.066 | -2.934 | -1.682 |
| psychological_mastery | 3.000 | 0.734 | -2.266 | -3.445 |
| psychological_curiosity | 2.330 | 0.000 | -2.330 | -6.960 |
| psychological_autonomy | 3.000 | -1.000 | -4.000 | -3.950 |
| psychological_immersion | 3.000 | 0.330 | -2.670 | -1.455 |
| functional_progress_feedback | 1.330 | 3.000 | +1.670 | 6.996 |
| functional_ease_of_control | 2.000 | 2.670 | +0.670 | 7.161 |
| functional_audiovisual_appeal | 3.000 | 1.000 | -2.000 | 0.569 |
| functional_goals_and_rules | 3.000 | 3.000 | +0.000 | 10.030 |
| functional_challenge | 3.000 | -1.599 | -4.599 | -1.599 |

### Uncharted (1659420)

- Steam name: UNCHARTED™: Legacy of Thieves Collection; match: `vetted_variant`; genre: Action adventure game; PXI samples: 1.
- Text: `VICReg_review/tags/game_descriptions/1659420.txt`; sentences: 28; chars: 3160.
- MAE: 1.588; RMSE: 1.735; functional MAE: 1.534; psychological MAE: 1.642.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 1.670 | 0.066 | -1.604 | -8.022 |
| psychological_mastery | 2.000 | 0.734 | -1.266 | -11.818 |
| psychological_curiosity | 1.000 | 0.000 | -1.000 | -8.363 |
| psychological_autonomy | 1.000 | -1.000 | -2.000 | -26.823 |
| psychological_immersion | 2.670 | 0.330 | -2.340 | -6.182 |
| functional_progress_feedback | 0.000 | 3.000 | +3.000 | 13.437 |
| functional_ease_of_control | 2.000 | 2.670 | +0.670 | 9.053 |
| functional_audiovisual_appeal | 3.000 | 1.000 | -2.000 | -0.675 |
| functional_goals_and_rules | 2.000 | 3.000 | +1.000 | 29.774 |
| functional_challenge | -1.000 | -2.000 | -1.000 | -16.291 |

### World of Tanks (1407200)

- Steam name: World of Tanks; match: `exact`; genre: Real time strategy; PXI samples: 5.
- Text: `VICReg_review/tags/game_descriptions/1407200.txt`; sentences: 27; chars: 2357.
- MAE: 0.628; RMSE: 0.914; functional MAE: 0.734; psychological MAE: 0.521.

| dimension | actual PXI mean | predicted mean | error | raw unclipped |
|---|---:|---:|---:|---:|
| psychological_meaning | 0.066 | 0.066 | -0.000 | -2.805 |
| psychological_mastery | 0.734 | 0.734 | +0.000 | -3.093 |
| psychological_curiosity | 0.400 | 0.000 | -0.400 | -6.792 |
| psychological_autonomy | 0.668 | -1.000 | -1.668 | -3.576 |
| psychological_immersion | 0.866 | 0.330 | -0.536 | -1.117 |
| functional_progress_feedback | 1.600 | 3.000 | +1.400 | 5.571 |
| functional_ease_of_control | 2.468 | 2.670 | +0.202 | 6.714 |
| functional_audiovisual_appeal | 1.000 | 1.000 | +0.000 | -0.213 |
| functional_goals_and_rules | 2.668 | 3.000 | +0.332 | 8.488 |
| functional_challenge | 1.200 | -0.538 | -1.738 | -0.538 |

## Caveats

- This report uses the requested real game text from `VICReg_review/tags/game_descriptions`, not the PXI pseudo_text baseline.
- The PXI head is still trained on only 21 overlapping games, so these numbers are a small-sample diagnostic rather than a stable benchmark.
- Large raw-unclipped ranges and clipped outputs indicate the current linear PXI head is still poorly calibrated for some real description texts.
- Several overlap matches are vetted title variants; they are listed in the per-game table under `match`.
