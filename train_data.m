% Machine Learning Homework Assignment 2
% Name :        Kshipra Avinash Kode
% Student ID :  ON08413
% Email :       kkode1@umbc.edu
% Function to train data

function [W, accuracy_vector] = train_data(W, images_train, labels_train, images_test, labels_test,test)
    accuracy_vector = zeros(1,50);
    for t = 1:5
        for i = (1:size(images_train,1))
            val = images_train(i,:)*W';
            if(val >= 0)
                val = 1;
            else 
                val = -1;
            end
            actual = labels_train(i);
            if actual == test
                actual = -1;
            else
                actual = 1;
            end
            if val ~= actual
                W = W + actual.*images_train(i,:);            
            end
            acc = test_data(images_test, labels_test, W, test);
            accuracy_vector = [accuracy_vector acc];
        end
    end 
end

