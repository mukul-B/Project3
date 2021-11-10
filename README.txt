build_model
1) model with random values is created having w1,b1,w2,b2 as directory is created
2) using sklearn onehot encoder labels is transformed to vector
3) the data dis split into batch size
4) and for each batch size activation of each node in neural network
5) similarly H and Z is calculated by using numpy matmul
6) y_prediction is calculated by using softmax of sklearn
7) model is updated by calling updateModel function
8) step 4 to 6 is repeated for all the batches
9) loss function is displayed is flag is true
10) step 3 to 9 is repeated for num_passes

updateModel
here the partial derivatives calculated using numpy
and then weights are baises are updated with given learning rate

calculate_loss
loss function is calculated after taking log of predicted and multiply with label vector , taking there sum using numpy
