# DeepSDF Shape representation
3D shape representation with DeepSDF


# Description

This small project shows how two 3D shapes (.off mesh files) can be voxelized and sdf values can be assigned to them.
Then the DeepSDF model is used to train these two shapes in an implicit neural representation. Doing so can provide a 
to reconstruct and generate shapes with one variable. 

# How to use?
Any two shapes (.off files) can be placed in [dataset](./dataset/) folder to let the DeepSDF have a transition from one shape to other with one single paramter.

To train the model, run [train](./src/train.py).

# Dataset
Any samples from ModelNet10 (with ```.off``` extension) can be used here. 
# Parts
The [config](./config.json) is used to put all the config info on it.


The DeepSDF model is placed into [this](./model/DeepSDFmodel.py). You can also see [notebook](./src/notebook.ipynb) file.

# Results
In the following, the training is done two times on two pairs of shapes. The first pair is a toilet-bed case and the second contains two chairs. In each case, the first sample is coded 1 and the second sample coded 2 initially. The following is showing the reconstruction of shapes with different codes. The ground truth are somehow shown when the code is 1 or 2. 

![image](images/image.png)

# Notes
DeepSDF is a great tool for implicit 3D shapre representation. However, the transition should be smarter than this to avoid meaningless shapes between these two shapes. 



