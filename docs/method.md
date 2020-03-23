## Method
ACDTE stands for Automatic Concept based Decision Tree Explanations. It is a framework to help domain expert, with little to no knowledge in machine learning, to be able to understand and trust deep learning models used in image classification. ACDTE only works with CNN used in image classification. Concept based explanations are preferred over saliency maps and other techniques since it offers a higher level of abstraction that provide the user with NLP rule explanations as well as their counterfactuals in terms of semantically meaningful concepts. ACDTE is also very flexible in terms of locality, meaning the user can control how semantically close the concepts used in the explanations are to the instance need to be explained.  
  
The pipeline of ACDTE consists of three major stages:
-	Automatic extraction of concepts
- Concepts linear models
- Building the decision tree and extracting the explanations
  
ACDTE is also modular since any stage can be replaced with alternatives that have the same functionality to allow for more control over the resulting explanations. For example, in stage 1 we can replace the segmentation model used (DeepLab v3+) with a better model or even a super pixel technique such as SLIC for faster performance.  
The whole pipeline is shown in the next figure  
  
![Pipeline](https://i.imgur.com/K5moKJ3.png)  
  
We start with image set that is used to extract the concepts from and use [segmentation.ipynb](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/segmentation.ipynb) to extract the images and masks using DeepLab v3+ segmentation model. We then cache the results for further use. We use the images and masks in all of the other notebooks to extract the segments from the images. We use K means clustering with K depending on the number of segments extracted from the dataset. After that, we filter the clusters formed using some heuristics and build linear regression pipelines that consists of PCA to first reduce the dimensionality followed by logistic regression. These linear models predict whether the corresponding concept is present in a query image or not. We convert the images to concept binary vector which is used in building the decision tree. Finally, we extract rules from the decision tree by following the path from the root to the leaves. We get the NLP labels of the concepts from the segmentation model which provides the segments labels as well. We use the majority label as the name of the concept.  