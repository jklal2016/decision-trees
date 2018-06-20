# decision-trees
Spark and Scala -- Decision Trees using Classification, Regression, and Logistic Regression

OBJECTIVE:

Decision Trees for Classification, Regression, and Logistic Regression over the HIGGS dataset. 

For each algorithm:
1. Use pipelines and cross-validation to find the best configuration of parameters and their performance. Use the same splits of training and test data when comparing performances between the algorithms.
2. Find which features are more relevant for classification or regression.
3. Provide training times in the cluster when using different cores.

RESULTS:

A smaller dataset was used called "HIGGS_r.csv.gz" because the larger dataset did not produce results. All features were used unless otherwise indicated. The execution time is the training time, done from classification until the training data is used.

For Classification:

1 node (in folder "Classification_all_features_1node"): Best Max Depth = 5, Best Impurity = gini, Best Max Bins = 20, The execution time is 199.965892073seconds., Accuracy = 0.68969448398665, Test Error = 0.31030551601335

5 nodes (in folder "Classification_all_features_5nodes"): Best Max Depth = 5, Best Impurity = entropy, Best Max Bins = 10, The execution time is 186.104289429seconds., Accuracy = 0.6835602679862066, Test Error = 0.3164397320137934

10 nodes (in folder "Classification_all_features_10nodes"): Best Max Depth = 5, Best Impurity = gini, Best Max Bins = 20, The execution time is 133.527966249seconds., Accuracy = 0.6855871034119388, Test Error = 0.31441289658806115

For Regression:

1 node (in folder "Regression_all_features_1node"): Best Max Depth = 10, Best Impurity = variance, Best Max Bins = 20, The execution time is 117.460050154seconds., Accuracy = 0.4474237841000399, Test Error = 0.5525762158999601

5 nodes (in folder "Regression_all_features_5nodes"): Best Max Depth = 10, Best Impurity = variance, Best Max Bins = 20, The execution time is 81.276392503seconds., Accuracy = 0.4465909085232288, Test Error = 0.5534090914767712

10 nodes (in folder "Regression_all_features_10nodes"): Best Max Depth = 10, Best Impurity = variance, Best Max Bins = 20, The execution time is 76.59614016seconds., Accuracy = 0.44753639132499884, Test Error = 0.5524636086750012

For Logistic Regression:

1 node (in folder "Logistic_all_features_1node"): Best Max Iter = 30, Best Reg Param = 0.1, Best Threshold = 0.25, The execution time is 344.345335325seconds., Accuracy = 0.6621190356660763, Test Error = 0.3378809643339237

5 nodes (in folder "Logistic_all_features_5nodes"): Best Max Iter = 50, Best Reg Param = 0.1, Best Threshold = 0.25, The execution time is 259.454261572seconds., Accuracy = 0.6627047378364921, Test Error = 0.3372952621635079

10 nodes (in folder "Logistic_all_features_10nodes"): Best Max Iter = 30, Best Reg Param = 0.1, Best Threshold = 0.25, The execution time is 239.463666155seconds., Accuracy = 0.6621329816257515, Test Error = 0.3378670183742485

For the relevant features for Regression (using 1 node):

Low features (first 21, in folder "Regression_low_features_1node"): Best Max Depth = 5, Best Impurity = variance, Best Max Bins = 5, The execution time is 122.533009335seconds., Accuracy = 0.4881053673485045, Test Error = 0.5118946326514955

High features (last 7, in folder "Regression_high_features_1node"): Best Max Depth = 10, Best Impurity = variance, Best Max Bins = 20, The execution time is 88.182617019seconds., Accuracy = 0.4492789296011011, Test Error = 0.550721070398899

All features: See above for Regression: 1 node.

Using only low features for Regression produces a higher accuracy and is, therefore, more relevant.
