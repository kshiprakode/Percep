% Machine Learning Homework Assignment 2
% Name :        Kshipra Avinash Kode
% Student ID :  ON08413
% Email :       kkode1@umbc.edu
% 
clear
load MNIST_digit_data;

rand('seed', 1);

% Loading train data
% Loading data for 1 and 6, 1000 per class.

indices = find(labels_train(:, 1) == 1);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_1 = images_train(indices, :);
labels_train_1 = labels_train(indices, :);
images_train_1 = images_train_1(1:1000, :);
labels_train_1 = labels_train_1(1:1000, :);

indices = find(labels_train(:, 1) == 6);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_2 = images_train(indices, :);
labels_train_2 = labels_train(indices, :);
images_train_2 = images_train_2(1:1000, :);
labels_train_2 = labels_train_2(1:1000, :);

images_train_complete = images_train;
labels_train_complete = labels_train;

images_train = [images_train_1 ; images_train_2];
labels_train = [labels_train_1 ; labels_train_2];

% Stores sorted train data for homework part 5g
images_train_sorted = [images_train_1 ; images_train_2];
labels_train_sorted = [labels_train_1 ; labels_train_2];

images_train_combined = zeros(size(images_train));

% Randomizing the train data
indices_loc = randperm(size(images_train,1));
for i = 1:2000
  images_train_combined(i,:) = images_train(indices_loc(i) , :);
end
labels_train_combined = labels_train(indices_loc);

images_train = images_train_combined;
labels_train = labels_train_combined;


% Loading test data
indices = find(labels_test(:, 1) == 1);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_test_1 = images_test(indices, :);
labels_test_1 = labels_test(indices, :);
images_test_1 = images_test_1(1:1000, :);
labels_test_1 = labels_test_1(1:1000, :);

indices = find(labels_test(:, 1) == 6);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_test_2 = images_test(indices, :);
labels_test_2 = labels_test(indices, :);
images_test_2 = images_test_2(1:958, :);
labels_test_2 = labels_test_2(1:958, :);

images_test = [images_test_1 ; images_test_2];
labels_test = [labels_test_1 ; labels_test_2];

images_test_combined = zeros(size(images_test));

% Randomizing test data
indices_loc = randperm(size(images_test,1));
for i = 1:size(images_test,1)
  images_test_combined(i,:) = images_test(indices_loc(i) , :);
end
labels_test_combined = labels_test(indices_loc);

images_test = images_test_combined;
labels_test = labels_test_combined;


W = zeros(1,size(images_train,2));
% Homework 2 , Solutions 5c

disp('---------------------------Homework 5c---------------------------');
[W, accuracy_values] = train_data(W, images_train, labels_train, images_test, labels_test,6);
figure();
disp('Plotting graph for Homework Solution 5c');
plot(accuracy_values);
axis([1,10000,0,1]);
title('Solution 5c: Accuracy vs iterations ');
ylabel('Accuracy');
xlabel('No. of Iterations');

disp('---------------------------Homework 5d---------------------------');
disp('Plotting graph for Homework Solution 5d');
% Homework 2 , Solutions 5d
C = W;
C(C < 0) = 0;
B = W;
B(B > 0) = 0;
figure();
imshow([reshape(C, [28 28]),reshape(-1.*B, [28 28])]);
title('Solution 5d');

disp('---------------------------Homework 5e---------------------------');
disp('Plotting visualizations for Homework Solution 5e');
% Best and worst images in 1
weights = zeros(1,size(images_test_1,1));
for i = 1:size(images_test_1,1)
    weights(i) = images_test_1(i,:)*W';    
end

[~, index] = sort(weights);
figure();
for i = 1:20
    subplot(1,20,i) , imshow(reshape(images_test_1(index(i),:), [28 28]));
end
title('Worst Images of 1');

figure();
j = 1;
for i = size(images_test_1,1) - 19:size(images_test_1,1)
    subplot(1,20,j) , imshow(reshape(images_test_1(index(i),:), [28 28]));
    j = j + 1;
end
title('Best Images of 1');

% Best and worst images in 6
weights = zeros(1,size(images_test_2,1));
for i = 1:size(images_test_2,1)
    weights(i) = images_test_2(i,:)*W'; 
end
[~, index] = sort(weights);

figure();
for i = 1:20
    subplot(1,20,i) , imshow(reshape(images_test_2(index(i),:), [28 28]));
end
title('Best Images of 6');

figure();
j = 1;
for i = size(images_test_2,1) - 19:size(images_test_2,1)
    subplot(1,20,j) , imshow(reshape(images_test_2(index(i),:), [28 28]));
    j = j + 1;
end
title('Worst Images of 6');


disp('---------------------------Homework 5f---------------------------');
disp('Plotting graph for Homework Solution 5f');
% Homework 2 , Solutions 5f

% Randomly flipping the data to add 10% train error
randomize = randperm(2000,200);
images_train_rand = images_train;
labels_train_rand = labels_train;

for i = randomize
   if(labels_train_rand(i) == 6)
       labels_train_rand(i) = 1;
   else
       labels_train_rand(i) = 6;
   end
end

W = zeros(1,size(images_train,2));
[W ,accuracy_values] = train_data(W, images_train_rand, labels_train_rand, images_test, labels_test,6);
figure();
plot(accuracy_values);
axis([1,10000,0,1]);
title('Solution 5f: Accuracy vs iterations on randomly flipped data');
ylabel('Accuracy');
xlabel('No. of Iterations');



disp('---------------------------Homework 5g---------------------------');
disp('Plotting graph for Homework Solution 5g');
% Homework 2 , Solutions 5g
W = zeros(1,size(images_train,2));
[W ,accuracy_values] = train_data(W, images_train_sorted, labels_train_sorted, images_test, labels_test,6);
figure();
plot(accuracy_values);
axis([1,10000,0,1]);
title('Solution 5g: Accuracy vs iterations on sorted train data');
ylabel('Accuracy');
xlabel('No. of Iterations');


disp('---------------------------Homework 5i a-------------------------');
% Homework 2, Solutions 5i
% Part A) Training only 10 images per class
% Loading train data
% Loading train data
% Loading data for 1 and 6, 1000 per class.

images_train = images_train_complete;
labels_train = labels_train_complete;

indices = find(labels_train(:, 1) == 1);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_1 = images_train(indices, :);
labels_train_1 = labels_train(indices, :);
images_train_1 = images_train_1(1:10, :);
labels_train_1 = labels_train_1(1:10, :);

indices = find(labels_train(:, 1) == 6);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_2 = images_train(indices, :);
labels_train_2 = labels_train(indices, :);
images_train_2 = images_train_2(1:10, :);
labels_train_2 = labels_train_2(1:10, :);

images_train = [images_train_1 ; images_train_2];
labels_train = [labels_train_1 ; labels_train_2];

% Stores sorted train data for homework part 5g
images_train_combined = zeros(size(images_train));

% Randomizing the train data
indices_loc = randperm(size(images_train,1));
for i = 1:20
  images_train_combined(i,:) = images_train(indices_loc(i) , :);
end
labels_train_combined = labels_train(indices_loc);

images_train = images_train_combined;
labels_train = labels_train_combined;

W = zeros(1,size(images_train,2));

[W, accuracy_values] = train_data(W, images_train, labels_train, images_test, labels_test,6);
figure();
disp('Plotting graph for Homework Solution 5i a');
plot(accuracy_values);
axis([1,200,0,1]);
title('Solution 5i: Accuracy vs iterations on 10 train data samples');
ylabel('Accuracy');
xlabel('No. of Iterations');

disp('---------------------------Homework 5i b-------------------------');
% Homework 2, Solutions 5i
% Part B) Training all images per class
% Loading train data

% Loading data for 1 and 6, 1000 per class.
images_train = images_train_complete;
labels_train = labels_train_complete;

indices = find(labels_train(:, 1) == 1);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_1 = images_train(indices, :);
labels_train_1 = labels_train(indices, :);

indices = find(labels_train(:, 1) == 6);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_2 = images_train(indices, :);
labels_train_2 = labels_train(indices, :);

images_train = [images_train_1 ; images_train_2];
labels_train = [labels_train_1 ; labels_train_2];

images_train_combined = zeros(size(images_train));

% Randomizing the train data
indices_loc = randperm(size(images_train,1));
for i = 1:size(images_train)
  images_train_combined(i,:) = images_train(indices_loc(i) , :);
end
labels_train_combined = labels_train(indices_loc);

images_train = images_train_combined;
labels_train = labels_train_combined;

W = zeros(1,size(images_train,2));

[W, accuracy_values] = train_data(W, images_train, labels_train, images_test, labels_test,6);
figure();
disp('Plotting graph for Homework Solution 5i b');
plot(accuracy_values);
title('Solution 5i: Accuracy vs iterations on all train data samples');
ylabel('Accuracy');
xlabel('No. of Iterations');


disp('---------------------------Homework 5h---------------------------');
% Homework 2, Solutions 5h
clear
load MNIST_digit_data;

rand('seed', 1);

% Loading train data
% Loading data for 2 and 8, 1000 per class.

indices = find(labels_train(:, 1) == 2);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_1 = images_train(indices, :);
labels_train_1 = labels_train(indices, :);
images_train_1 = images_train_1(1:1000, :);
labels_train_1 = labels_train_1(1:1000, :);

indices = find(labels_train(:, 1) == 8);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_train_2 = images_train(indices, :);
labels_train_2 = labels_train(indices, :);
images_train_2 = images_train_2(1:1000, :);
labels_train_2 = labels_train_2(1:1000, :);

images_train = [images_train_1 ; images_train_2];
labels_train = [labels_train_1 ; labels_train_2];

% Stores sorted train data for homework part 5g
images_train_sorted = [images_train_1 ; images_train_2];
labels_train_sorted = [labels_train_1 ; labels_train_2];

images_train_combined = zeros(size(images_train));

% Randomizing the train data
indices_loc = randperm(size(images_train,1));
for i = 1:2000
  images_train_combined(i,:) = images_train(indices_loc(i) , :);
end
labels_train_combined = labels_train(indices_loc);

images_train = images_train_combined;
labels_train = labels_train_combined;


% Loading test data
indices = find(labels_test(:, 1) == 2);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_test_1 = images_test(indices, :);
labels_test_1 = labels_test(indices, :);
images_test_1 = images_test_1(1:1000, :);
labels_test_1 = labels_test_1(1:1000, :);

indices = find(labels_test(:, 1) == 8);
indices_loc = randperm(size(indices,1));
indices = indices(indices_loc);
images_test_2 = images_test(indices, :);
labels_test_2 = labels_test(indices, :);
images_test_2 = images_test_2(1:974, :);
labels_test_2 = labels_test_2(1:974, :);

images_test = [images_test_1 ; images_test_2];
labels_test = [labels_test_1 ; labels_test_2];

images_test_combined = zeros(size(images_test));

% Randomizing test data
indices_loc = randperm(size(images_test,1));
for i = 1:size(images_test,1)
  images_test_combined(i,:) = images_test(indices_loc(i) , :);
end
labels_test_combined = labels_test(indices_loc);

images_test = images_test_combined;
labels_test = labels_test_combined;


W = zeros(1,size(images_train,2));
% Homework 2 , Solutions 5h c

disp('---------------------------Homework 5h c-------------------------');
[W, accuracy_values] = train_data(W, images_train, labels_train, images_test, labels_test,8);
figure();
disp('Plotting graph for Homework Solution 5h c');
plot(accuracy_values);
axis([1,10000,0,1]);
title('Solution 5h c: Accuracy vs iterations ');
ylabel('Accuracy');
xlabel('No. of Iterations');

disp('---------------------------Homework 5h d-------------------------');
disp('Plotting graph for Homework Solution 5h d');
% Homework 2 , Solutions 5h d
C = W;
C(C < 0) = 0;
B = W;
B(B > 0) = 0;
figure();
imshow([reshape(C, [28 28]),reshape(-1.*B, [28 28])]);
title('Solution 5h d');

disp('---------------------------Homework 5h e-------------------------');
disp('Plotting visualizations for Homework Solution 5h e');
% Best and worst images in 2
weights = zeros(1,size(images_test_1,1));
for i = 1:size(images_test_1,1)
    weights(i) = images_test_1(i,:)*W';    
end

[~, index] = sort(weights);
figure();
for i = 1:20
    subplot(1,20,i) , imshow(reshape(images_test_1(index(i),:), [28 28]));
end
title('Worst Images of 2');

figure();
j = 1;
for i = size(images_test_1,1) - 19:size(images_test_1,1)
    subplot(1,20,j) , imshow(reshape(images_test_1(index(i),:), [28 28]));
    j = j + 1;
end
title('Best Images of 2');

% Best and worst images in 8
weights = zeros(1,size(images_test_2,1));
for i = 1:size(images_test_2,1)
    weights(i) = images_test_2(i,:)*W'; 
end
[~, index] = sort(weights);

figure();
for i = 1:20
    subplot(1,20,i) , imshow(reshape(images_test_2(index(i),:), [28 28]));
end
title('Best Images of 8');

figure();
j = 1;
for i = size(images_test_2,1) - 19:size(images_test_2,1)
    subplot(1,20,j) , imshow(reshape(images_test_2(index(i),:), [28 28]));
    j = j + 1;
end
title('Worst Images of 8');

