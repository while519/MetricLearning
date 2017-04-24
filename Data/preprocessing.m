function [ Y ] = preprocessing(X, out_dim, varargin)
%% preprocessing - preprocess the data matrix to denoise and compress the information
% 
% Y = preprocessing(X, out_dim, varargin);
%
%   X - (M x N) document-term matrix
%   out_dim - scalar
%   varargin - string or cell
%
% Returns :
%
%   Y - (M x out_dim) compact data matrix
%
% Description :
%   This m-file function computes the preprocessing step for the input
%   content data. It essentially performs a dimensionality reduction step
%   so that the obtained data representation has fewer features than its number
%   of samples.
%
% Example : Y = preprocessing(X, 50, 'PCA');

%%
%
% Author   : Yu Wu
%            University of Liverpool
%            Electrical Engineering and Electronics
%            Brownlow Hill, Liverpool L69 3GJ
%            yuwu@liv.ac.uk
% Last Rev : Tuesday, February 14, 2017 (GMT) 10:49 AM
% Tested   : Matlab_R2016a
%
% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.
%
% Fixed by GTM+0 (1/17/14) to work for xxx
% and to warn for xxx.  Also ensures that 
% output is all xxx, and allows the option of forcing xxx  

%% 
% initialise
if ~exist('out_dim', 'var')
    out_dim = 100;
    type = 'PCA';
    para = [];
end

if length(varargin) == 0
    type = 'PCA';       % default values
    para = [];
elseif length(varargin) == 1
    type = varargin{1};
    para = [];
elseif length(varargin) == 2
    type = varargin{1};
    para = varargin{2};
end

%%
% PCA step
switch lower(type)
    case 'pca'
        % normalize the data
        X = X - min(X(:));
        X = X/max(X(:));
        X = bsxfun(@minus, X, mean(X, 1));
        
        % determining the rank of matrix X
        r = rank(X);
        if r < out_dim
            warning(['The target dimensionality is too big that the feaures' ...
                ' are not independent set, we use output dim = ' num2str(r) ' instead']);
        end
        
        [P, Lambda] = eig(X'*X);
        [lambda, ind] = sort(diag(Lambda), 'descend');
        P = P(:, ind(1 : out_dim));
        Y = bsxfun(@minus, X, mean(X,1)) * P;
        
        % Plot the proportion of the variance
        figure
        plot(lambda(1 : out_dim)./sum(lambda), '*:', 'MarkerSize', 10);
        title('PCA: variance measured by the eigenvalues');
        
    case 'lda'      % dirichlet allocation
        [WS, DS] = SparseMatrixtoCounts(X');
        % hyperparameters setting
        T = out_dim;     % number of topics
        BETA = 0.01;
        ALPHA = 50 / T;
        N = 300;
        SEED = 3;   % random seed
        OUTPUT = 1; % what output to show(0 = no output; 1 = iterations; 2 = all output)
        
        % this function might need a few minutes to finish
        tic
        [WP,DP,~ ] = GibbsSamplerLDA( WS , DS , T , N , ALPHA , BETA , SEED , OUTPUT );
        toc
        
        % visualize the results
        figure
        imagesc(WP);
        title('Word topic matrix structure');

        figure
        imagesc(DP);
        title('Document topic matrix structure');
        Y = full(bsxfun(@rdivide, DP, sum(DP,2)));
        
        
    otherwise
        error(['unspecified input arguments' type]);
end

end

