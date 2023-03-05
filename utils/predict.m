function error = predict(X, labels)
prob = mysoftmax(X);
predictions = gather(prob) ;
[~,pre_label] = sort(predictions, 'descend') ;
error = sum(~bsxfun(@eq, pre_label(1,:)', labels)) ;

