# Stacked denoising autoencoder


Implements stacked denoising autoencoder in Keras without tied weights.


To read up about the stacked denoising autoencoder, check the following paper:
    

Vincent, Pascal, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. 
"Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." 
Journal of Machine Learning Research 11, no. Dec (2010): 3371-3408.

    
Requirements:


1. Python 3.4
2. keras 1.2
3. numpy
4. scipy

This architecture can be used for unsupervised representation learning in varied domains, including textual and structured data. The following paper uses this stacked denoising autoencoder for learning patient representations from clinical notes, and thereby evaluating them for different clinical end tasks in a supervised setup:

[Madhumita Sushil, Simon Šuster, Kim Luyckx, Walter Daelemans. "Patient representation learning and interpretable evaluation using clinical notes." Journal of Biomedical Informatics, Volume 84 (2018): 103-113](https://www.sciencedirect.com/science/article/pii/S1532046418301266)
