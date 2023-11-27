% ��˹������ʵ�ֲ��裺
% 1. ��j��ͼ�����һ���˲�������˹����������ø�˹��ͨ�˲���;
% 2. ��j��ͼ����в���2���²����������õ� j-1 ��ͼ��;
% 3. ����1,2�����е���������ֱ�������0��ͼ�������

function output = gaussian_pyramid(image, level)

h =1/16 * [1, 4, 6, 4, 1];
filt = h' * h;

% filt = 1/256 * [ 1  4  6  4 1;      % ��˹�ں�
%                 4 16 24 16 4;
%                 6 24 36 24 6;
%                 4 16 24 16 4;
%                 1  4  6  4 1 ];
             
output{1} = imfilter(image, filt, 'replicate', 'conv');
temp_img = image;

for i = 2 : level
    temp_img = temp_img(1: 2: end, 1: 2: end);  %���²�������Сͼ�񣬲���Ϊ2
    output{i} = imfilter(temp_img, filt, 'replicate', 'conv'); 
end

end
