# TGBAD
Chengru Yu, **Chang Liu***, Mingcheng Yang, and Yilin Xu,
[A Tensor-based Granular Ball Model for Anomaly Detection](paper/2026-TGBAD.pdf), 2026, Code: [https://github.com/from-china-to/TGBAD](https://github.com/from-china-to/TGBAD).

## Abstract
Tensor-based multi-view clustering transforms multi-view data from the object level to the clustering level, whereas anomaly detection requires a continuous ranking of anomaly scores for individual original samples. This difference in the granularity of data representation makes it challenging to directly integrate the two. Replacing samples with granular-ball centers can reduce the quantity of processing units; however, without explicit member decoding, ball-level structural evidence still cannot be directly associated with the original records. To resolve this limitation, this study puts forward a Tensor-based Granular Ball Model for Anomaly Detection (TGBAD), which applies the tensor graph learner to granular-ball centers, preserves the membership relationships between samples and granular balls, and maps the ball-level reconstruction residuals back to individual samples. Subsequently, the multi-tensor granular-ball score, feature-tail rarity, and Isolation Forest scores are fused in a label-free manner using average rank percentiles and a fixed Top-2 rule. On 16 public datasets, TGBAD significantly outperforms 9 of the 10 baselines in terms of AUC; scalability experiments further validate the efficiency of granular execution. The code is publicly available online at https://github.com/from-china-to/TGBAD.

## Usage
Install the dependencies, then run `TGBAD.py`. With no arguments it runs the built-in `Example.mat` demo (`Example.mat` contains two 4-dimensional samples, `0000` and `1111`):

```
pip install -r requirements.txt

cd code
python TGBAD.py
```

You can get outputs as follows:

```
==================================================================
TGBAD: Tensor-based Granular-Ball Anomaly Detection
==================================================================
Data file : Example.mat
Data      : n=2 samples, d=4 features
Granules  : 2 granular balls, 2 tensor units, ADMM iterations=13 (converged)
Runtime   : 2.37s
------------------------------------------------------------------
Anomaly scores (in [0, 1]; higher = more anomalous):
  sample    1: gbs=0.000  tail=0.500  iso=0.500  ->  score = 0.5000
  sample    2: gbs=1.000  tail=0.500  iso=0.500  ->  score = 0.7500  <-- anomaly
Ranking   : most anomalous -> sample 2 > sample 1
------------------------------------------------------------------
AUC=1.0000  PR-AUC=1.0000  F1-Max=1.0000   (ground-truth labels used for evaluation only)
==================================================================
```

Run on any of the 16 public datasets (a `trandata` or `X`/`Y` .mat file):

```
python TGBAD.py ../data/annthyroid.mat
python TGBAD.py ../data/annthyroid.mat --out scores.npy
```

Ground-truth labels are used only for offline evaluation, never for fitting.

## Citation
If you find TGBAD useful in your research, please consider citing:

```
@article{yu2026tensor,
  title={A Tensor-based Granular Ball Model for Anomaly Detection},
  author={Yu, Chengru and Liu, Chang and Yang, Mingcheng and Xu, Yilin},
  year={2026},
  note={Code available at https://github.com/from-china-to/TGBAD}
}
```

## Contact
If you have any questions, please contact liuchangai@alu.scu.edu.cn.
