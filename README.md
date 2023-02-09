## Do Invariances in Deep Neural Networks Align with Human Perception?

Code to repreoduce experiments for AAAI-2023 paper: https://arxiv.org/abs/2111.14726

1. ``PerceptualSimilarity`` is a slightly modified version of https://github.com/richzhang/PerceptualSimilarity.  Install it in your virtualenv using ``python setup.py install``

2. Code to generate IRIs: ``invariances_in_reps/human_nn_alignment/reg_free_loss.py``

3. Code to train SimCLR: ``invariances_in_reps/human_nn_alignment/deep-learning-base/simclr_training.py``

4. For plots and tables: ``invariances_in_reps/human_nn_alignment/measure_alignment.ipynb``

## Citation

```
@inproceedings{nanda2023invariances,
    title={Do Invariances in Deep Neural Networks Align with Human Perception?},
    author={Nanda, Vedant and Majumdar, Ayan and Kolling, Camilla and Dickerson, John P. and Gummadi, Krishna P. and Love, Bradley C. and Weller, Adrian},
    booktitle={AAAI},
    year={2023}
}
```
