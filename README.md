## Model 
Consider the following signal plus noise model
<img src="https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BX%7D%20%3D%20%5Csum_%7BI%3D1%7D%5Ek%20%5Clambda%20%5Cmathbf%7Bu%7D%5Cmathbf%7Bv%7D%5E%5Ctop%20%2B%20W">
where W's each entry is iid Gaussian or W's distribution of orthogonally invariant.

## TODO
- [x] include Gaussian AMP
- [ ] write the procedures to normalize the matrix
- [ ] write sample usage of this package
- [ ] check for AMP improvment for svd # test for different noise, how much is the improvements. # my pot

### Data Transformation


### Sample Usage


## TODO
1. Unfinished part in pca analysis, what are the standard procedures? Transform and impute? what should we observe? How to select signals?

2. Joint EB


# log return for each day...
1. Should we renormalize it?
2. How to choose PC?
3. 

# TODO
1. to do on the gaussian noise, use the original signal solver. # # my pot first # chang pot

# TODO 

1. check of solve_signal. (We might don't have ground truth for signal) # thing first, chang pot.
3. general AMP. # chang pot.
4. check for AMP improvment for svd # test for different noise, how much is the improvements. # my pot


Compare for the subspace spaned by actual pcs and estimated pcs



## TODO 

1. current implementation dimension is skewed. 

2. over iterations sigma will get the numerical instability

3. try for the orthogonal amp, use only the last f.
4. high dimension em estimation
4.1 Bivariate empirical bayes
4.2 multivariate amp
4.3 For checking the method, Look into the cross distribution 

5. even though the alignment is already high, denoising still might help.

1. 改algo
1.1 high dimension.
1.2 gai  n support; 
2. visualization

3. 看别的data set （可能就不搞）


1. 改成在只用最后一个f
2. 一维度update，降低mle的nsupport
2. 改prior estimate

两个错
gaussian
1. sqrt_root
number of iterations 5.