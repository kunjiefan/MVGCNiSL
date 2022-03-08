# MVGCN-iSL
 THis is the code for our work "Multi-view graph convolutional network for predicting cancer cell-specific synthetic lethality". Our model, MVGCN-iSL, comprises three parts. In the first, the GCN processes multiple biological networks independently as cell-specific and cell-independent input graphs to obtain graph-specific representations that provide diverse information for SL prediction. In the second part, a max pooling operation integrates several graph-specific representations into one, and in the third part, a multi-layer deep neural network (DNN) model utilizes these integrated representations as input to predict SL.


## run the model
```
python main.py
```
By default, the model runs on the 'K562' cell line using all five graph features and four types of omics features as the input.      
Please refer to "main.py" for a list of parameters to be adjusted.
