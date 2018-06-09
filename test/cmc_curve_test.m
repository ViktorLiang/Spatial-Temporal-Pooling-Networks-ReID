num_test = 20;
num_class = 10;

match_scores = zeros(length(num_test),length(num_class));  
true_labels = zeros(length(num_test),length(num_class));  
for i=1:length(num_test)  
   for j=1:length(num_class)  
       [x,y]=find(num_class(j)==num_train);  

       %选取匹配程度的中值  
       label_distances(i,j) = median(match_dist(i,y));  
       if num_test(i)==num_class(j)  
           true_labels(i,j)=1;  
       end  
   end  
end  

%生成CMC  
max_rank = length(num_class);  

%Rank取值范围  
ranks = 1:max_rank;  

%排序  
label_distances_sort = zeros(length(num_test),length(num_class));  
true_labels_sort = zeros(length(num_test),length(num_class));  
for i=1:length(num_test)  
   [label_distances_sort(i,:), ind] = sort(label_distances(i,:));  
   true_labels_sort(i,:) =  true_labels(i,ind);  
end  

%迭代  
rec_rates = zeros(1,max_rank);  
tmp = 0;  
for i=1:max_rank  
   tmp = tmp + sum(true_labels_sort(:,i));  
   rec_rates(1,i)=tmp/length(num_test);  
end  