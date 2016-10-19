% get ground truth densityies for each image
% density for each pixel = 1/area of bbox

clc
clear

load('data/Cam181/01/01/bbox.mat');
n = 500;
gtDensities0 = cell(n,1);
bboxTrueCount = zeros(500,1);
% miny  = 5;
% minx  = 5;
% maxy = 235;
% maxx = 347;

% set(gcf, 'Color', 'white');
% for i = 1:n
%     im = imread(['data/551Lane/' num2str(i+500) '.jpg']);
%     image(im);
%     for j = 1:size(BBOXFG{i},1)
%         box = BBOXFG{i}(j,:);
%         linewidth = 1;
%         rectangle('Position', RectLTRB2LTWH(box), 'LineWidth', linewidth, 'EdgeColor','g');
%     end
%     saveas(gcf, ['data/551Lane/' 'label' num2str(i) '.png']);
% %     imwrite(gcf,'2.png')
% end

for i=1:n
    box = bbox{i};   %x1,y1, x2(col),y2(row)
    m = size(box,1);
    gtDen0 = zeros(240,352);

    for j = 1:m
        x1 = min(max(1,box(j,1)),352); 
        y1 = min(max(95,box(j,2)),240);  % 95 is the height of the mask(cutting line). Pay attention!!! 95 or 1???
        x2 = min(max(1,box(j,3)),352);
        y2 = min(max(95,box(j,4)),240);
        s = (y2 - y1) * (x2 - x1);
        den0 = 1/s;
        Dtemp0 = zeros(240,352);
        Dtemp0(y1:y2-1, x1:x2-1) = den0;  % Pay attention! Not Dtemp0(y1:y2, x1:x2) if y1= y2=95, Dtemp0 will be all 0
        gtDen0 = gtDen0 + Dtemp0;  
    end
    gtDensities0{i} = gtDen0;
    bboxTrueCount(i,1) = sum(sum(gtDensities0{i}));
end

save('data/Cam181/gtDensities_BBOXori.mat','gtDensities0');




