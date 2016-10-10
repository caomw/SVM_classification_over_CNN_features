% -------------------------------------------------------------------------
% SVM Training with Pegasos - Cite the following
% ---------------------------
% @article{shalev2011pegasos,
% title={Pegasos: Primal estimated sub-gradient solver for svm},
% author={Shalev-Shwartz, Shai and Singer, Yoram and Srebro, Nathan and Cotter, Andrew},
% journal={Mathematical programming},
% volume={127},
% number={1},
% pages={3--30},
% year={2011},
% publisher={Springer} 
% } 
% ---------------------------
% This is a simplified MATLAB implementation of their C++ code available at 
% http://www.cs.huji.ac.il/~shais/code/
% ---------------------------
% INPUTS : 
% (a) X as N x d matrix, N = number of examples, d = dim of each example
% (b) Y as N x 1 vector, contains class labels {1,-1} for each example
% ---------------------------
% OUTPUTS : 
% (a) W as 1 x d vector of the weights trained 
% (b) b as 1 x 1 bias trained 
% ---------------------------
% Author : Sukrit Shankar 
% -------------------------------------------------------------------------
function [W,b] = pegasosSVMTrain(X,Y)

% Specify N = No of examples, d = dimension of each example from the input matrices
N = size(X,1);  d = size(X,2); 

% Set the configuration parameters
lambda = 1; % Parameter in the algo
k = ceil (0.1 * N); % Parameter in the algo 
maxNumIterations = 3000; % Max iterations for convergence
tolerance = 10^(-3); % For the norm of the difference between W vectors within consecutive iterations

% Initialize the to-be-taken mean vector to output W
w = rand(1,d);
w = w / (sqrt(lambda) * norm(w));

% Start the training loop 
for i = 1:1:maxNumIterations
    
    % Apply the real algorithm 
    b = mean(Y - X * w(i,:)');
    indices = randi([1,N],k,1); 
    tempX = X(indices,:);
    tempY = Y(indices,:);
    indicesRefined = (tempX * w(i,:)' + b) .* tempY < 1;
    eta = 1 / (lambda * i);
    wUpdate = (1 - eta * lambda) * w(i,:) + ...
         (eta / k) * sum(tempX(indicesRefined,:) ...
         .* repmat(tempY(indicesRefined,:),1,size(tempX,2)),1);
    
    % Get w for the next iteration - useful in calculating the norm
    w(i+1,:) = min(1,1/(sqrt(lambda) * norm(wUpdate))) * wUpdate;
    
    % Check for tolerance 
    if(norm(w(i+1,:) - w(i,:)) <= tolerance)  
        break;
    end
    
    % Clear variables for the loop
    clear indices tempX tempY indicesRefined eta wUpdate; 
    
    % Print the progress
    fprintf('\n Performing Pegasos Iteration = %d (%d) [Norm of difference = %f]'...
        ,i,maxNumIterations,norm(w(i+1,:) - w(i,:))); 
end

% Say if the algorithm could converge in the max iterations or not
if(i >= maxNumIterations) % Did not break from the loop before reaching max iterations
    fprintf ('\n ------------------------------------------'); 
    fprintf('\n Pegasos NOT converged in %d iterations',maxNumIterations);
    fprintf ('\n ------------------------------------------');  
else
    fprintf ('\n ------------------------------------------'); 
    fprintf('\n Pegasos WELL converged in %d iterations',i);
    fprintf ('\n ------------------------------------------');
end

% Write the variables to return 
W = mean(w,1);
b = mean(Y - X * W');



