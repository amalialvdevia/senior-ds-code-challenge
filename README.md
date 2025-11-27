# senior-ds-code-challenge
This repository contains the complete solution to the technical challenge for the Senior Data Scientist role, focused on predicting product returns in an e-commerce environment and optimizing intervention decisions using a business-aligned financial metric.

1. Baseline and limitations

The provided baseline model uses logistic regression with a fixed threshold of 0.5.
Although it achieves relatively high accuracy, in practice it predicts almost all orders as non-returns, resulting in:

Precision = 0

Recall = 0

Business value = 0 USD per order

ROC-AUC ≈ 0.56

This behavior is typical in imbalanced datasets: accuracy alone does not reflect the true business impact, especially when false negatives (FN) generate significant financial losses.

2. Defining a business-aligned metric

To evaluate the model appropriately, I defined a financial metric based on the cost matrix:

Cost of a return: 18 USD

Cost of a preventive intervention: 3 USD

Net benefit of a successful intervention: +15 USD per TP

Cost of an unnecessary intervention: −3 USD per FP

The optimized metric is:

𝐸
𝑉
𝑜
𝑟
𝑑
𝑒
𝑟
=
15
⋅
𝑇
𝑃
−
3
⋅
𝐹
𝑃
𝑁
EV
order
	​

=
N
15⋅TP−3⋅FP
	​


This allows comparing models and thresholds not only on statistical performance but on actual dollar impact.

3. Improved model and threshold optimization

A stronger model was developed using:

Random Forest,

One-Hot Encoding for categorical variables,

A full preprocessing + modeling pipeline,

Stratified cross-validation to avoid overfitting.

On the test set, the improved model achieves:

ROC-AUC ≈ 0.62

At the optimal threshold τ ≈ 0.40 (chosen by maximum financial gain):

Precision ≈ 0.31

Recall ≈ 0.80

F1 ≈ 0.44

The model’s expected value is:

≈ 1.65 USD per order,

Which translates into ≈ 165,000 USD in monthly savings for 100,000 orders.

4. Conclusion

While the baseline achieves superficially strong accuracy, it delivers zero economic value.
The proposed approach integrates:

a financial cost-benefit metric,

threshold optimization,

a robust non-linear model,

resulting in a substantial improvement in both predictive performance and business impact.

This solution is fully reproducible and ready to support real-world decision-making in a production environment.
