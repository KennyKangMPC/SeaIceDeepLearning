
%I = imread('test.jpg');
I = imread('icefloe.png');
I = rgb2gray(I);
figure, imshow(I);
[r, c] = size(I);

%% Otsu

t = graythresh(I);
bw = im2bw(I, t);
figure,imshow(bw);

p = find(bw == 1);
ic = length(p) / (r * c);

% im = double(I);
% n = 0;
% for i = 1 : r
%     for j = 1 : c
%         if im(i, j) >  t * 255
%             n = n + 1;
%         end
%     end
% end
% ic = n / (r * c);


%% multi-Otsu

N = 2;
thresh = multithresh(I, N);
seg = imquantize(I, thresh);
figure, imshow(seg,[])

for i = 1 : (N + 1)
    p = find(seg == i);
    coverage(i) = length(p) / (r * c);
    
    for j = 1 : length(p)
        s(j) = I(p(j));
    end
    average_intensity(i) = sum(s(:)) / length(p);
end




