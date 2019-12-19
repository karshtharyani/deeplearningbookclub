function updateParameters(regularization_method, w1,b1,w2,b2,w3,b3)

switch regularization_method
    case 'none'
        w3 = w3 - lr_w*grad3_w;
        b3 = b3 - lr_b*grad3_b;
        w2 = w2 - lr_w*grad2_w;
        b2 = b2 - lr_b*grad2_b;
        w1 = w1 - lr_w*grad1_w;
        b1 = b1 - lr_b*grad1_b;
    case 'l2'
        w3 = w3 - lr_w*grad3_w + ;
        b3 = b3 - lr_b*grad3_b;
        w2 = w2 - lr_w*grad2_w;
        b2 = b2 - lr_b*grad2_b;
        w1 = w1 - lr_w*grad1_w;
        b1 = b1 - lr_b*grad1_b;
    case 'l3'
        w3 = w3 - lr_w*grad3_w;
        b3 = b3 - lr_b*grad3_b;
        w2 = w2 - lr_w*grad2_w;
        b2 = b2 - lr_b*grad2_b;
        w1 = w1 - lr_w*grad1_w;
        b1 = b1 - lr_b*grad1_b;
    otherwise
        w3 = w3 - lr_w*grad3_w;
        b3 = b3 - lr_b*grad3_b;
        w2 = w2 - lr_w*grad2_w;
        b2 = b2 - lr_b*grad2_b;
        w1 = w1 - lr_w*grad1_w;
        b1 = b1 - lr_b*grad1_b;
        
        

end


function value = calculate_l2()

    
end

function value = calculate_l1()

end