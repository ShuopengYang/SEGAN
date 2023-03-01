# SEGAN
Procedures and instructions of SEGAN

Dependencies :
Python >= 3.7 (Recommend to use Anaconda)
PyTorch >= 1.7

Illustration :
Use the corresponding pre-trained model to generate pressure fine grid data using pressure coarse grid data and permeability fine grid data. We upload the pre-training 
model to Releases

Inference :
python /inference_SEGAN.py -iPCoarse inputs_PCoarse -iKRefined inputs_KRefined 
