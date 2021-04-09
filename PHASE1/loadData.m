%
% Example: loadData('data/abalone.data');
%
function [data, labels] = loadData(filename)
  file = importdata(filename);
  
  if isstruct(file)
      data = file.data;
      labels = cell2mat(file.textdata);
  else
      data = file(:, 2:size(file, 2));
      labels = file(:, 1);
  end
end

