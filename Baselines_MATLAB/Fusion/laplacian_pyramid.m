% ������˹�����������ã��ؽ���˹��������
function output = laplacian_pyramid(image, level)

h = 1/16 * [1, 4, 6, 4, 1];
filt = h'* h;

output{1} = image;
temp_img = image;

for i = 2 : level   % �˲����²���
    temp_img = temp_img(1: 2: end, 1: 2: end);
    output{i} = imfilter(temp_img, filt, 'replicate', 'conv'); 
end

% ����Ԥ��в�ؽ����ͼ���ȥԭʼ�ĵ�j������ͼ��
% ����i��Ϊ��˹������i���� i+1 �㾭���ڲ�Ŵ��ͼ��Ĳ�
for i = 1 : level - 1
   [m, n] = size(output{i});
   output{i} = output{i} - imresize(output{i + 1}, [m, n]); 
end

end
