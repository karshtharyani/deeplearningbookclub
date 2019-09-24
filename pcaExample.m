function pcaExample

    image = double(imread(fullfile(matlabroot, 'toolbox', 'images', 'imdata', 'coloredChips.png')));
    [m,n,p] = size(image);
    data = reshape(image, m*n, p);

    [coeff, score, eigvals] = pca(data);

    transformedData = data*coeff;

    for idx = 1:p
        imagePC(:,:, idx) = reshape(transformedData(:,idx), m, n);
        subplot(1,p,idx);
        imshow(imagePC(:,:,idx));
    end
end