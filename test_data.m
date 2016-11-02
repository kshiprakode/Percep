% Machine Learning Homework Assignment 2
% Name :        Kshipra Avinash Kode
% Student ID :  ON08413
% Email :       kkode1@umbc.edu
% Function to test data

function accuracy = test_data(images_test, labels_test, W, test)
    miscount = 0;   
    for i = 1:size(images_test,1)
        predict = images_test(i,:)*W';
        
        if predict > 0
            predict = 1;
        else 
            predict = -1;
        end
        
        label = labels_test(i);
        if label == test
            label = -1;
        else 
            label = 1;
        end
        if predict ~= label
            miscount = miscount + 1;
        end
    end
    accuracy = 1 - miscount/size(images_test,1);
    
end


