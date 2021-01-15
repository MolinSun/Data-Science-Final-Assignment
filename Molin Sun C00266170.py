#!/usr/bin/env python
# coding: utf-8

# # Data Science & Artificial Intelligence Final Examination Alternative Continuous Assessment
# ## Student Name: Molin Sun
# ## Student Number: C00266170

# ## 1 Data Pre-processing
# ### 1.1 Introduction
# #### 1.1.1 Purpose
# * Data preprocessing is to process raw data 
# * In the real world,data is durty. They are always incomplete,noisy and inconsistent.
#   * Incomplete: lacking attribute value
# * Data preprocessing is to make these durty data easier to be interpreted by the algorithm.
# 
# #### 1.1.2 Quality data
# ### 1.2 Task of preprocessing
# #### 1.2.1 Data cleaning
# #### 1.2.2 Data integration
# #### 1.2.3 Data transformation
# #### 1.2.4 Data reduction
# #### 1.2.5 Data discretisation
# ### 1.3 Implementation
# ### 1.4 Application
# ### 1.5 Conclusion
# ### 1.6 Presentation

# ## 2 Decision Trees
# ### 2.1 Introduction
# * Decision trees are a supervised learning method that creates a tree-like model to divide a large heterogeneous data set into smaller data sets based on different conditions.
# * Decision trees are used for classification and regression
# * The goal of Decision trees is to create a model that can predict the value of the target variable by learning decision rules.
# * The figure shown below is an example of decision tree
# ![avatar](Desktop\DecisionTree.png)
# 
# * For this  simple example, it uses some rules to predict the species of animals. The model uses 3 attributes from the data set, namely feather, fly and finn.
# * As shown above, a decision tree is drawn upside down with its root at the top. The sentences with question marks represent conditions or internal node to split a node into branches. And the end of branch is leaf node representing class label.
# 
# ### 2.2 ID3 algorithm
# #### 2.2.1 Introduction
# * Is an algorithm used to generate a decision tree from a data set in a decision tree. 
# * Is used to select the best attributes which can best split the data set.
# * Is a greedy algorithm.
# * The steps of the algorithm:
#  * select the best attribute using the decreasing speed of information entropy as the standard 
#  * Use the selected best attributes as the decision node and use the different value of the attributes to split remaining instances
#  * Repeat the previous step recursively for each child node
#  * If all instances are perfectly classified, stop the iteration.
# 
# #### 2.2.2 Entropy
# * Entropy is a measure of order in a system. The more ordered a system is, the lower entropy the system has.
#   * Expression: Entropy(S) = ∑(i=1 to l)-|Si|/|S| * log2(|Si|/|S|)
#   * S is the set of examples
#   * Si is a subset of S whose value is vi under the target attribute
#   * l is the size of the range of the target attribute
# * Example
#   * In the example dataset, there is one special attribute(whether the person will buy the computer) and four regular attributes(age, income, student and credit).
# ![avatar](Desktop\Table.png)
#   * Calculate entropy : as shown in the table, there are 14 examples, 9 positive and 5 negatives.So the entropy of S relative to this classification is 
#     * Entropy(S) = -(9/14)log2(9/14)-(5/14)log2(5/14) = 0.9403
#   * Calculate entropy of other attributes using expression :
#    * ∑(i=1 to k)|Si|/|S|Entropy(Si)  (k refers to the range of the attribute we are testing)
#     * Age
#      * Entropy(young) = -(2/5)log2(2/5)-(3/5)log2(3/5) = 0.970
#      * Entropy(medium) = -(4/4)log2(4/4)-(0/4)log2(0/4) = 0 
#      * Entropy(old) = -(3/5)log2(3/5)-(2/5)log2(2/5) = 0.970
#      * Entropy(Age|S) = 5/14*Entropy(young) + 4/14*Entropy(medium)+5/14*Entropy(old) = 0.6935
#     * Income
#      * Entropy(high) = -(2/4)log2(2/4)-(2/4)log2(2/4) = 1
#      * Entropy(medium) = -(4/6)log2(4/6)-(2/6)log2(2/6) = 0.9183 
#      * Entropy(low) = -(3/4)log2(3/4)-(1/4)log2(1/4) = 0.8113
#      * Entropy(Income|S) = 4/14*Entropy(high) + 6/14*Entropy(medium)+4/14*Entropy(low) = 0.9111
#     * Student
#      * Entropy(yse) = -(6/7)log2(6/7)-(1/7)log2(1/7) = 0.5917
#      * Entropy(no) = -(4/7)log2(4/7)-(3/7)log2(3/7) = 0.9852
#      * Entropy(Student|S) = 7/14*Entropy(yes) + 7/14*Entropy(no) = 0.7885
#     * Credit
#      * Entropy(good) = -(6/8)log2(6/8)-(2/8)log2(2/8) = 0.8113
#      * Entropy(excellent) = -(3/6)log2(3/6)-(3/6)log2(3/6) = 1 
#      * Entropy(Credit|S) = 8/14*Entropy(good) + 6/14*Entropy(excellent)=0.8922
#      
# #### 2.2.3 Information Gain
# * The best attribute has the greatest reduction in entropy. So information gain is the expected reduction in entropy using A attribute to split the data set.
#   ![image-2.png](attachment:image-2.png)
# * The expression shown above equal: Gain(S，A) = Entropy(S) - Entropy(A|S)
# * Calculate Information gain(using the example mentioned above)
#   * Gain(S,Age)= Entropy(S) - Entropy(Age|S) = 0.9403 - 0.6935 = 0.2468
#   * Gain(S,Income)= Entropy(S) - Entropy(Income|S) = 0.9403 - 0.9111 = 0.0292
#   * Gain(S,Student)= Entropy(S) - Entropy(Student|S) = 0.9403 - 0.7885 = 0.1518
#   * Gain(S,Credit)= Entropy(S) - Entropy(Credit|S) = 0.9403 - 0.8922 = 0.0481
# * As a result, age is the best attribute to split the data set.
# 
# ### 2.3 Implementation

# ### 2.4 Application
# ### 2.5 Conclusion
# * Advantages
#  * Decision trees is easy to understand, interpret and visualize
#  * Work well on discrete and categorical variables
# * Disadvantages
#  * When a learner creates a over-complex tree, which will lead to overfitting.
# * Recommendations for future learning
# 
# ### 2.6 Presentation

# ## 3 Bayesian Classifier
# ### 3.1 Introduction
# ### 3.2 How does Bayesian Classifier work
# ##### 3.2.1.1 Probabilities
# ##### 3.2.1.2  Conduct classifier
# ### 3.3 Implementation
# ### 3.4 Application
# ### 3.5 Conclusion
# ### 3.6 Presentation

# ## 4 K-Nearest  Neighbour Algorithm
# ### 4.1 Introduction
# ### 4.2 How does K-NN work?
# #### 4.2.1 How to choose the value of K 
# #### 4.2.2 Euclidean distance
# #### 4.2.3 K-NN algorithm
# ### 4.3 Implementation
# ### 4.4 Application
# ### 4.5 Conclusion
# ### 4.6 Presentation

# ## 5 Support vector machine
# ### 5.1 Introduction
# ### 5.2 How does SVM work
# #### 5.2.1 Linear Separators
# #### 5.2.2 Margin
# ### 5.3 Implementation
# ### 5.4 Application
# ### 5.5 Conclusion
# ### 5.6 Presentation

# ## 6 References
