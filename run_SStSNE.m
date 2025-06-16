function run_SStSNE(X_csv, label_csv, dim, thresh_cat, out_csv)
%   L  : n*1 matrix where L(i)is the class of sample X(i,:)
%        if L(i) == -1, X(i,:)is an unlabeled point.
%   dim : embedding dimensionality (scalar integer; 2)
%   thresh_cat : ([0.5,1[) Percentage of points of the same class in
%                the soft neighborhood of a labeled point.
    addpath('Semi-supervised.t-SNE-main/');
    X = readmatrix(X_csv);
    L = readmatrix(label_csv);
    Y = SStSNE(X, L, str2num(dim), str2num(thresh_cat), 50, 1e-4, false, 2);
    writematrix(Y, out_csv);
end