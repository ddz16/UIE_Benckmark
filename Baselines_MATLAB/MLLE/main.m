clc
clear 
close all
addpath('Method');
Image_dir = 'challenging-60';
listing = cat(1, dir(fullfile(Image_dir, '*.png')));
% The final output will be saved in this directory:
result_dir = fullfile(Image_dir, 'result');
% Preparations for saving results.
if ~exist(result_dir, 'dir'), mkdir(result_dir); end

for i_img = 1:length(listing)
    Input = imread(fullfile(Image_dir,listing(i_img).name));
    [~, img_name, ~] = fileparts(listing(i_img).name);
    img_name = strrep(img_name, '_input', '');     
    %% �?部自适应的颜色校�? 
    Out1 = LACC(Input);    
    %% �?部自适应的颜色校�? 
    Result = (LACE(Out1));
    %% Resize�?256*256
    Result = imresize(Result, [256 256]);
    imwrite(Result, fullfile(result_dir, [img_name, '.png'])); 
end


