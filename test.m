%load('sido0')
% A: data matrix,  n x d
% b: label, n x 1, 1 or -1
% idx = randperm(length(data_news));
%data_news = data_news(idx,:);
num_user = max(data_news(:,1));
num_item = max(data_news(:,2));
%data(:,3) = data(:,3)/max(data(:,3));
num_feat = num_user + num_item;
% A=sparse(length(data_news),num_feat+1);
% for i=1:length(data_news)
%     A(i,data_news(i,1)) = 1;
%     A(i,data_news(i,2)+num_user) = 1;
% 	A(i,num_feat+1) = 1;
% end
col_a=[[1:size(data_news,1)]';[1:size(data_news,1)]'];
col_b=[data_news(:,1);data_news(:,2)+num_user];
A=sparse(col_a,col_b,ones(size(data_news,1)*2,1));
A=[A,ones(size(data_news,1),1)];
b = data_news(:,3);
% A = data_news;
% b = label_news;
model.grad = @(z) squared_error_1 (z);%logistic_1 (z);%
%for m=10:10:50
opts.m = 10;
% idx = randperm(length(A));
% A = A(idx,:);
data.A = A;
data.b = b;
[n, d] = size(A);

%opts.eta = 1e-2; [error1, cost1] = AFTRL(data, model, opts);
%opts.eta = 1e-1; [error2, cost2] = AFTRL(data, model, opts);
% opts.eta = 10; [error1, cost1] = AFTRL(data, model, opts);
% 
% opts.m=15;
% opts.eta = 10; [error2, cost2] = AFTRL(data, model, opts);
% 
% opts.m=20;
opts.eta = 0.01; %[error3, cost3] = AFTRL(data, model, opts);
%[error, cost, w, BT_P, BT_N, norms, preds] = AFTRL(data, model, opts);
%[error, cost, w, BT_P, BT_N, norms] = AFTRL_reg(data, model, opts);
[error, cost, BT_P, BT_N, norms] = AFTRL_reg_CCFM(data, model, opts);
%end
% 
% opts.m=25;
% opts.eta = 10; [error4, cost4] = AFTRL(data, model, opts);
% 
% opts.m=30;
% opts.eta = 10; [error5, cost5] = AFTRL(data, model, opts);


%opts.eta = 10; [error4, cost4] = AFTRL(data, model, opts);
%opts.eta = 100; [error5, cost5] = AFTRL(data, model, opts);
% figure;
% step = floor(n/20);
% plot(error1(step:step:n) / n, 'r-x', 'LineWidth', 1.8); hold on;
% plot(error2(step:step:n) / n, 'g-o', 'LineWidth', 1.8); hold on;
% plot(error3(step:step:n) / n, 'b-<', 'LineWidth', 1.8); hold on;
% plot(error4(step:step:n) / n, 'k-s', 'LineWidth', 1.8); hold on;
% plot(error5(step:step:n) / n, 'm-.', 'LineWidth', 1.8); hold on;
% legend('eta_1','eta_2','eta_3','eta_4','eta_5')
