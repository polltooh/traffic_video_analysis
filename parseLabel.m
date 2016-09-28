
clc;
clear;

imdir = '04/';

n = 500;
TureCount = zeros(n,1);
bbox = cell(500,1);
VType = cell(500,1);
for i = 1:n
   
    name = sprintf('%06d',i);
    s = xml2struct([imdir name '.xml']);
    
    c = size(s.annotation.vehicle,2);
    TureCount(i,1) = c;
    
    box = zeros(c,4);
    type = zeros(c,1);
    
    for j = 1:c
        xmin = s.annotation.vehicle{1,j}.bndbox.xmin.Text;
        ymin = s.annotation.vehicle{1,j}.bndbox.ymin.Text;
        xmax = s.annotation.vehicle{1,j}.bndbox.xmax.Text;
        ymax = s.annotation.vehicle{1,j}.bndbox.ymax.Text;     
        
        xmin = str2double(xmin);
        ymin = str2double(ymin);
        xmax = str2double(xmax);
        ymax = str2double(ymax);
        box(j,:)= [xmin, ymin, xmax, ymax];
        type(j,1) =str2double(s.annotation.vehicle{1,j}.type.Text);
    end
    
    bbox{i,1} = box;
    VType{i,1} = type;
    i
end

save([imdir 'TureCount.mat'], 'TureCount');
save([imdir 'bbox.mat'], 'bbox');
save([imdir 'VType.mat'], 'VType');
