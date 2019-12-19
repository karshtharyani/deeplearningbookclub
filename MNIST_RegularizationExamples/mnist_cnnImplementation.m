
% to reproduce the same result
rng(0);

% load the data
% if ~exist('data', 'var')
%     data = loadMNISTImages('data/train-images.idx3-ubyte');
%     labels = loadMNISTLabels('data/train-labels.idx1-ubyte') + 1;
%     tdata = loadMNISTImages('data/t10k-images.idx3-ubyte');
%     tlabels = loadMNISTLabels('data/t10k-labels.idx1-ubyte') + 1;
%     
    rp = randperm(size(trainImageVecs, 2));
    trainImageVecs = trainImageVecs(:, rp(1:10000));
    trainLabels = trainLabels(rp(1:10000));
    
    trp = randperm(size(testImageVecs, 2));
    testImageVecs = testImageVecs(:, trp(1:1000));
    testLabels = testLabels(trp(1:1000));
% end

% network setting
[~, num_data] = size(trainImageVecs);
[~, num_tdata] = size(testImageVecs);
num_epoch = 100;
batch_size = 100;
dim = 28;
tx = reshape(testImageVecs, dim, dim, []);
num_output = 10;
dim_filter = 9;
dim_pool = 2;
num_filter = 15;

% basic setting
wc = 1e-1 * randn(dim_filter, dim_filter, num_filter);
bc = zeros(num_filter, 1);

%%%% FILL THE BLANK
% we will apply the (dim_filter x dim_filter) filter to the image
% the result should be (conv_dim x conv_dim) map.
% also, if this map is operated by pooling, it will be (out_dim x out_dim) map.
conv_dim = 20;
out_dim = 10;
%%%%

num_hidden = out_dim^2 * num_filter;

r  = sqrt(6) / sqrt(num_hidden + num_output);
w = rand(num_output, num_hidden) * 2 * r - r;
b = zeros(num_output, 1);

lr = 2;
lr_c = 1e-2;
gam = 0.5;

% loss/acc history
loss_list = zeros(num_epoch, 1);
acc_list = zeros(num_epoch, 1);

% we will use mean pool filter
pool_filter = ones(dim_pool) / dim_pool^2;
conv_idx = 1 : dim_pool : conv_dim-dim_pool+1;

for i = 1 : num_epoch
    rp = randperm(num_data);
    disp(i)
    for j = 1 : batch_size : (num_data - batch_size + 1)
        % random batch
        x = reshape(trainImageVecs(:, rp(j:j+batch_size-1)), dim, dim, []);
        l = trainLabels(rp(j:j+batch_size-1));
        
        % feature map
        map = zeros(conv_dim, conv_dim, num_filter, batch_size);
        map_pooled = zeros(out_dim, out_dim, num_filter, batch_size);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % feed forward by convolution and pooling
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for p = 1 : batch_size
            im = x(:, :, p);
            
            for q = 1 : num_filter
                filter = squeeze(wc(:, :, q));
                
                %%%% FILL THE BLANK
                % get the map created by applying convolution with the filter.
                % this map will be activated with the internal_sigmoid function,
                % after added with the bias.
                % replace ? with your code.
                convim = internal_sigmoid( conv2(im,filter,'valid') + bc(q));
                %%%%
                
                map(:, :, q, p) = convim;
                
                %%%% FILL THE BLANK
                % get the pooled map created by applying mean pooling.
                % this is just same as applying conv2 with the pool filter.
                fmap = conv2(map(:, :, q, p),pool_filter);
                map_pooled(:, :, q, p) = fmap(conv_idx, conv_idx);
                %%%%
            end
        end
        
        % get the score
        map_pooled = reshape(map_pooled, [], batch_size);
        s = exp(w * map_pooled + repmat(b,1,100));
        
        %%%% FILL THE BLANK
        % each row vector should indicate the probability for each digit
%         s = internal_sigmoid(s);
        for as=1:100
            s(:,as) = s(:,as)/sum(s(:,as));
        end
        %%%%
        
        % compute the error
        I = sub2ind(size(s), l', 1:batch_size);
        p = zeros(size(s));
        p(I) = 1; % Assign 1 only on where 'labels' indicate
        err = (s - p)/batch_size;
        
        %%%%%%%%%%%%%%%%%%
        % back propagation
        %%%%%%%%%%%%%%%%%%
        
        grad_w = err * map_pooled';
        grad_b = sum(err, 2);
        
        err_pool = reshape(w'*err, [], out_dim, num_filter, batch_size);
        err = zeros(conv_dim, conv_dim, num_filter, batch_size);
        
        % Upsample the error
        for p = 1 : batch_size
            for q = 1 : num_filter
                err(:, :, q, p) = kron(err_pool(:, :, q, p), pool_filter);
            end
        end
        
        err = err .* map .* (1-map);
        
        % Gradient w.r.t to the weight term for each filter
        grad_wc = zeros(size(wc));
        for q = 1 : num_filter
            grad = zeros(size(grad_wc, 1));
            for p = 1 : batch_size
                grad = grad + conv2(x(:, :, p), rot90(err(:, :, q, p), 2), 'valid');
            end
            grad_wc(:, :, q) = grad;
        end
        
        grad_bc = zeros(size(bc));
        % Gradient w.r.t to the bias term for each filter
        for q = 1 : num_filter
            err_b = err(:, :, q, :);
            grad_bc(q) = sum(err_b(:));
        end
        
        %%%% FILL THE BLANK
        % update the weights
        w = w - lr*grad_w;
        b = b - lr*grad_b;
        wc = wc-lr_c*grad_wc;
        bc = bc-lr_c*grad_bc;
        %%%%
    end
    
    % decay
    if ~mod(i, 40)
        lr = lr * gam;
        lr_c = lr_c * gam;
    end
    
    %%%% FILL THE BLANK
    % record the training loss and accuracy
    loss_list(i) = -sum(log(s(I)))/batch_size;
    [~, pred_labels] = max(s);
    acc_list(i) = length(find(pred_labels == l'))/batch_size;
    
    
    %%%%
end

% get the test output
map = zeros(conv_dim, conv_dim, num_filter, num_tdata);
map_pooled = zeros(out_dim, out_dim, num_filter, num_tdata);

for p = 1 : num_tdata
    im = tx(:, :, p);
    
    for q = 1 : num_filter
        filter = squeeze(wc(:, :, q));
        %%%% FILL THE BLANK
	% replace ? with your code
        convim = internal_sigmoid(conv2(im,filter,'valid') + bc(q));
        map(:, :, q, p) = convim;
        fmap = conv2(map(:, :, q, p),pool_filter);
        map_pooled(:, :, q, p) = fmap(conv_idx, conv_idx);
        %%%%
    end
end

%%%% FILL THE BLANK
% get the test score
% each row vector indicates the probability for each digit
map_pooled = reshape(map_pooled, [], num_tdata);
ts = exp(w * map_pooled + repmat(b,1,1000));
% ts = internal_sigmoid(ts);
for as=1:1000
    ts(:,as) = ts(:,as)/sum(ts(:,as));
end
%%%%

%%%% FILL THE BLANK
% record the test loss and accuracy
I = sub2ind(size(ts), testLabels', 1:num_tdata);
tloss = -sum(log(ts(I)))/num_tdata;
[~, tpred_labels] = max(ts);
tacc = length(find(tpred_labels == testLabels'))/1000;
disp(tacc)
%%%%

plot(1:num_epoch, loss_list, 'b-');
xlabel('Iteration'); ylabel('training loss'); figure;
plot(1:num_epoch, acc_list, 'b-');
xlabel('Iteration'); ylabel('training accuracy');