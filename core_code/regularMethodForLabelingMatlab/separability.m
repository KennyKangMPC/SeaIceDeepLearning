
clear all; clc;

k = 108;

I = imread('ch3ice.jpg');
I = rgb2gray(I);                                       
[m, n] = size(I);

%% thresholding
bw = zeros(size(I));
pp = find(I > k);
bw(pp) = 1;
IC = length(pp) / (m*n);
figure, imshow(bw, [])


%% separability

fxy = imhist(I, 256);
p = fxy / (m*n);

mg = 0;
for i = 1 : length(p)
    mg = mg + i*p(i);
end

sigma2_g = 0;
for i = 1 : length(p)
    sigma2_g = sigma2_g + ((i-mg)^2)*p(i);
end

m = 0; p0 = 0;
for i = 1 : k
    m = m + i*p(i);
    p0 = p0 + p(i);
end
sigma2_b = (mg*p0 - m)^2 / (p0*(1-p0));

eta = sigma2_b / sigma2_g;



