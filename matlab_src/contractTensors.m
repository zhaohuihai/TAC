function [X, numindX] = contractTensors(X, numindX, indX, Y, numindY, indY, varargin)
%* function [X, numindX] = contractTensors(X, numindX, indX, Y, numindY, indY, order)
%* Contraction of index indX of tensor X with index indY of tensor Y (X and Y have a number
%* of indices corresponding to numindX and numindY, respectively)
%===========================================================================================
%* example:
%* A(a,b,c,d) B(e,b,c,f)
%* AB(a,d,e,f) = sum{b,c}_[A(a,b,c,d)*B(e,b,c,f)]
%* AB = contractTensors(A, 4, [2, 3], B, 4, [2, 3])
%*
%* AB(d,e,a,f) = sum{b,c}_[A(a,b,c,d)*B(e,b,c,f)]
%* AB = contractTensors(A, 4, [2, 3], B, 4, [2, 3], [2, 3, 1, 4]) ;
%===========================================================================================
Xsize = ones(1, numindX);
Xsize(1:length(size(X))) = size(X);
Ysize = ones(1, numindY);
Ysize(1:length(size(Y))) = size(Y);

indXl = 1:numindX;
indXl(indX) = [];
indYr = 1:numindY;
indYr(indY) = [];
sizeXl = Xsize(indXl); % outer index dimension
sizeX = Xsize(indX); % inner index dimension
sizeYr = Ysize(indYr); % outer index dimension
sizeY = Ysize(indY); % inner index dimension
if prod(sizeX) ~= prod(sizeY)
    error('indX and indY are not of same dimension.');
end
%---------------------------------------------------------
if isempty(indYr)
    if isempty(indXl)
        X = permute(X, [indX]);
        X = reshape(X, [1, prod(sizeX)]);
        Y = permute(Y, [indY]);
        Y = reshape(Y, [prod(sizeY), 1]);
        
        X = X * Y ;
        
        Xsize = 1;
        return;
    else
        X = permute(X, [indXl, indX]);
        X = reshape(X, [prod(sizeXl), prod(sizeX)]);
        Y = permute(Y, [indY]);
        Y = reshape(Y, [prod(sizeY), 1]);
        
        X = X * Y ;
        
        Xsize = Xsize(indXl);
        X = reshape(X, [Xsize, 1]);
        return
    end
end
X = permute(X, [indXl, indX]);
X = reshape(X, [prod(sizeXl), prod(sizeX)]);
Y = permute(Y, [indY, indYr]);
Y = reshape(Y, [prod(sizeY), prod(sizeYr)]);

X = X * Y ;

Xsize = [Xsize(indXl), Ysize(indYr)];
numindX = length(Xsize);
X = reshape(X, [Xsize, 1]);
%--------------------------------------------------------------------
if nargin == 7
    order = varargin{1} ;
    X = permute(X, order) ;
end