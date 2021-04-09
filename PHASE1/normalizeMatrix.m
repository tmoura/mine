function [norm_data] = normalizeMatrix(oldVector)

[r,c] = size(oldVector);

norm_data = zeros(r,c);

for i = 1 : c
    maxError = max(oldVector(:,i));
    minError = min(oldVector(:,i));

    if (maxError == minError)
        norm_data(:,i) = oldVector(:,i);
    else
        norm_data(:,i) = (oldVector(:,i) - minError) / ( maxError - minError);
    end
end

end