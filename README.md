# Big Data Assignment № 2. Spark

Epinions.com was an online marketplace, where users posted customer reviews for the goods. When reading each other’s comments they were able to “trust” each other. It could be represented as a social graph, where each node represents a user and an oriented edge represents “trust”. We would like to create a recommendation system on this graph. For each user (node), we want to recommend 10 other users (nodes) to “trust”.

node2vec is the model that could be used to estimate the probability of an edge in the graph. We will take 10 most probable edges for the current node that are not presented in the graph, and use them as our recommendations. One of the methods to estimate the quality of recommendations is Mean Average Precision (MAP).

#Tasks: 
1- Implement a model node2vec
2- Implement training procedure in Scala
3- Implement evaluation procedure in Scala or Python

#Team:
Farah Atif
Yan Konyshev
