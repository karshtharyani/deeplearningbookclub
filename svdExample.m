function svdExample

% Image compression using SVD

image = imread(fullfile(matlabroot, 'toolbox', 'images', 'imdata', 'coloredChips.png'));
grayIm = rgb2gray(image);
grayIm = im2double(grayIm);

[U,S,V] = svd(grayIm);

% remove the last N values
n = [10, 100, 250];

figure('Name', 'singlular value and the image')
ax1 = subplot(1,numel(n)+1,1);
imshow(grayIm)
title(ax1, ['original with ' num2str(size(S,1)) ' singular value']);

for idn = 1:numel(n)
    Snew = S;
    Snew(end-n(idn):end, end-n(idn):n(idn)) = 0;
    newIm(:,:,idn) = U*Snew*V';
    
    % plot
    ax = subplot(1,numel(n)+1,idn+1);
    imshow(newIm(:,:,idn));
    title(ax, ['Omitted singular values: last ' num2str(n(idn)) '/' num2str(size(S,1))]);
end

end