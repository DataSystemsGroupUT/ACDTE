## Experiments Results
In all experiments, we used a subset of ADE20K dataset to extract concepts from and explain some of its instances. We also used pre-trained DeepLab v3+ model trained on ADE20K with Xception backend. For the model to be explained we used a ResNet50 model pre-trained on Places365 dataset. In order to have a more efficient pipeline, we used [segmentation.ipynb](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/segmentation.ipynb) to segment the images and cache the images with their masks for further usage in the other experiments. 
* [Selecting Extraction Layer](#selecting-extraction-layer)
* [Analyzing Extracted Concepts](#analyzing-extracted-concepts)
	* [Concepts Meaningfulness](#concepts-meaningfulness)
	* [Concepts importance](#concepts-importance)
* [Analyzing Decision Tree Performance](#analyzing-decision-tree-performance)
	* [Decision Tree Depth](#decision-tree-depth)
	* [Decision Tree Vs Random Forest](#decision-tree-vs-random-forest)
* [Counterfactual examples](#counterfactual-examples)

### Selecting Extraction Layer:
In this experiment, we select the best intermediate layer to extract the activation feature maps from based on the average accuracy of the linear models on a validation set. This reflects how well the linear models can detect a concept found in the images to be explained. The results can be seen in the next figure ![extraction_layer](https://i.imgur.com/DCd129t.png)  
It is clear from the accuracy that the linear models can detect the presence and absence of the corresponding concepts. We can also see an increase in the accuracy as the extraction layer is deeper. This behavior can be explained by the way we generate the segments since we use DeepLab v3+ which extracts highly semantic segments and thus the deeper layers are better at making sense of these segments.  
The code can be found [here](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/selecting_extraction_layer.ipynb)  
  
### Analyzing Extracted Concepts:
In this experiment, we show that the concepts extracted are meaningful as well as have various importance in separating the images in the given explanation set.  
#### Concepts Meaningfulness:
In the next image we can see some segments of the concept bed as well as the concept mountain the images are obtained by choosing the closest 4 images to the center of the cluster  
**Concept of a bed**
![bed](https://i.imgur.com/JCBCOOQ.png)  

**Concept of a mountain**
![mountain](https://i.imgur.com/nBFGBQL.png)  
The images show that the concepts are consistent and are semantically sound. These images can be extracted from the [visualization.ipynb](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/Visualization.ipynb).  
#### Concepts importance:
In this experiment, we validate that different concepts have different importance for the explanation set. We first define the concept importance by dividing its within-cluster distance by the total number of segments in the cluster, thus more important clusters have lower score. We then train a decision tree multiple times while removing the important concepts one by one. We do the experiment twice, once we remove the important concepts first and in the other we remove the important concepts last. We also repeat the same experiment while adding the concepts to the features used by the decision tree. In all experiments, we measure the fidelity using accuracy of the decision tree. The results can be seen in the next figures  
![concepts_importance](https://i.imgur.com/S92yXRO.png)  
The plots show the variable importance of concepts where the behavior of the plot shows that adding important concepts first leads to a faster increase in the fidelity score compared to adding less important concepts first. The same can be observed in the case of removing concepts. The code for this experiment can be found in [concept_importance.ipynb](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/concept_importance.ipynb).  
Another version of this experiment can be found in [locality_adding_concepts.ipynb](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/locality_adding_concepts.ipynb). We do the same adding concepts experiment but with a smaller neighborhood parameter which effectively make the concepts more compact and tightly related to the instance to be explained. The result in this experiment also shows the same behavior.  

### Analyzing Decision Tree Performance:
In the next experiments we analyze the performance of the decision tree using the fidelity measure.  
#### Decision Tree Depth:
In this experiment, we increase the depth of the decision tree and measure the fidelity using accuracy, we also include the test set accuracy as a measure of generalization capability.  
![depth_acc](https://i.imgur.com/FVv50HW.png)
The plot shows that as the depth increase the fidelity increases greatly till it reaches an accuracy of about 95% at the depth of only 20. This shows that the tree is very faithful to the original model and that its performance on the explanation set is similar to that of the original model. The test accuracy also shows that the decision tree has good generalization capability which means that the decision tree can be used as an interpretable surrogate model in the vicinity of the instance to be explained. The code can be found in [depth_vs_acc.ipynb](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/depth_vs_acc.ipynb)  
We also repeated the experiments with different neighborhoods in [locality_depth.ipynb](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/locality_depth.ipynb)  
#### Decision Tree Vs Random Forest:
In this experiment, we compare the performance of the decision tree built on the concept vectors with a decision tree built on the original activation maps as well as random forest built on both vectors. the results show that the decision tree built on the concept vectors is better than that built on the activation vector while its performance is close to the random forest in both cases  
```
For decision tree on concept vectors:
	train accuracy: 0.9373134328358208 +- 0.012822458697716078

For random forest on concept vectors:
	train accuracy: 0.9764650432050277 +- 0.002113454418953676

For decision tree on original vectors:
	train accuracy: 0.8929615082482324 +- 0.043437583077835774

For random forest on original vectors:
	train accuracy: 1.0 +- 0.0
```
  
### Counterfactual examples:
In this experiment, we built a decision tree found in next figure  
![Dectision_tree](https://i.imgur.com/aP3dNux.png)  
We then printed the rule for an instance which came out as  
```
instance number 31 is bathroom
decision id node 0 : (mountain (= 0) <= 0.5)
decision id node 1 : (bed (= 0) <= 0.5)
decision id node 2 : (sink (= 1) > 0.5)
```
Finally, we printed the counterfactual rule which turned out to be 
```
counter example class is living_room
decision id node 0 : (mountain <= 0.5)
decision id node 1 : (bed <= 0.5)
decision id node 2 : (sink <= 0.5)
```
The code can be found in [counterfactual.ipynb](https://github.com/DataSystemsGroupUT/ACDTE/blob/master/experiments/counterfactual.ipynb)
