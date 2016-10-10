% -------------------------------------------------------------------------
% Predicting labels {1,-1} from the SVM trained [W,b] 
% ---------------------------
% INPUTS : 
% (a) X as N x d matrix, N = number of test examples, d = dim of each example
% (b) W as 1 x d trained SVM vector 
% (c) b as 1 x 1 trained SVM bias 
% ---------------------------
% OUTPUTS : 
% (a) Y as N x 1 vector of test example predictions in {1,-1}
% ---------------------------
% Author : Sukrit Shankar 
% -------------------------------------------------------------------------
function Y = SVMTest(X,W,b)

% Do the fundamental SVM thing 
Y = sign(X * W'+ b);