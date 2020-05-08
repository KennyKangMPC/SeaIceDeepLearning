clc; clear all; close all;
I = imread('2.jpg');
I = rgb2gray(I);

figure,imshow(I);
[r, c] = size(I);
n_r = 2;
n_c = 3; % number of blocks
c_r = r/n_r; c_c = c/n_c;
t1 = (0:n_r-1)*c_r + 1; t2 = (1:n_r)*c_r;
t3 = (0:n_c-1)*c_c + 1; t4 = (1:n_c)*c_c;

figure,
for i = 1 : n_r
    for j = 1 : n_c
        temp = I(t1(i):t2(i), t3(j):t4(j));
        
        t=graythresh(temp);
        
        th = t*255;      
        thresh((i-1)*n_c+j) = th;
        
        [r_t,c_t] = size(temp);
        n = 0;
        for r1 = 1:r_t;
            for c1 = 1:c_t
                if temp(r1,c1)>th
                   n = n+1;
                end
            end
        end
        
        num((i-1)*n_c+j) = n;

        temp = im2bw(temp, t);
        IC_local((i-1)*n_c+j)=n/(r_t*c_t);
        
        ic0 = IC_local((i-1)*n_c+j) * 100;
                
        subplot(n_r, n_c, (i-1)*n_c+j);
%          figure,
        imshow(temp, 'border', 'tight');
        title({['{\it IC}=',num2str(ic0),'%'];['Threshold=',num2str(th)]}); 
%         pause(0.1);
hold on;
axis off;
    end
end

IC = sum(num(:)) / (r*c);

