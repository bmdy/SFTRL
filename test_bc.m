for i=1:length(data_news)
    data_news(i,:)=data_news(i,:)/norm(data_news(i,:));
end
A = [data_news,ones(length(data_news),1)];
b = label_news;

%for m=10:10:50
model.grad = @(z) logistic_1 (z);%squared_error_1 (z);%
opts.m = 25;
data.A = A;%(1:1000,:);
data.b = b;%(1:1000);
[n, d] = size(A);

opts.eta = 0.25; %[error3, cost3] = AFTRL(data, model, opts);
%[error, cost, w, BT_P, BT_N, norms] = AFTRL_reg(data, model, opts);
[error, cost, BT_P, BT_N, norms, preds] = AFTRL_CCFM(data, model, opts);
%end