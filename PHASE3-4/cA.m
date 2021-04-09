function ndata = cA(thirdDimension)
    
    ndata = [];
    
    tamThirdDimension = size(thirdDimension,3);
    
    for i = 1 : tamThirdDimension
        ndata = [ndata;thirdDimension(:,:,i)];
    end
end