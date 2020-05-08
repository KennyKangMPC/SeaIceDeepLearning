function bw1 = GVF_distance( I, sigma, GradientOn, GVFOn, Num, mu,...
    iter, alpha, beta, gamma, kappa, Dmin, Dmax, Ra_min, Ra, Rc, Rl, se, timer)
% Snake GVF with automatical initial contour based on local minima derived 
% from distance transform
% 
% Input data:
% I:               input color image
% sigma = 0:       gaussianBlur
% GradientOn = 1:  1 : Gradient on  0 : Gradient off
% 
% GVFOn = 1:       1 : GVF  0 : SVF
% Num:             number of GVF iterations
% mu = 0.1:        parameter for GVF (more noise, increase mu)
% 
% iter:            number of Snake iterations 
% alpha = 0.05:    internal weight that control snake's tension
% beta = 0:        internal weight that control snake's rigidity
% gamma = 1:       step size in one iteration
% kappa = 0.6:     external force weight
% Dmin = 0:        min raster of the snake
% Dmax = 1:        max raster of the snake
% 
% Ra_min = 20;     minmum area
% Ra:              maximum area
% Rc:              convexity threshlod, <= 1
% Rl:              threshold ratio between length and width, >= 1
% 
% se:              morphology structure element
% 
% timer:           iteration time
% 
% Output data:
% bw1:             output binary image 


I = rgb2gray(I);
bw = im2bw(I, graythresh(I));
[s1, s2] = size(bw);
s_1 = [0, s1, s1, 0];
s_2 = [0, 0, s2, s2];  % vertex of image boundary


% blured and draw the picture
if sigma ~= 0 
   f = gaussianBlur(I,sigma);
else
   f = I;
end;

% graient on the image in order to get edges
if GradientOn 
	f2 = abs(gradient2(double(f)));
else 
    f2 = f;
end;

% vector field
if GVFOn                          
    % GVF field
    [u, v] = GVF(f2, mu, Num);   % gradient vector force
else
    % standard vector field
    [u, v] = gradient2(f2);   % calculates standard external force vector
                              % field using gradient
end

% normalize vectors in vector field
mag = sqrt(u.*u + v.*v);
px = u ./ (mag + 1e-10);
py = v ./ (mag + 1e-10);

bw1 = bw;
for time = 1 : timer
    [label, num] = bwlabel(bw1, 4);
    a = zeros(num, 1);
    rc = zeros(num, 1);
    l = zeros(num, 1);
    w = zeros(num, 1);
    for n = 1 : num
        aa = regionprops(label == n, 'Area');   % area
        a(n) = cat(1, aa.Area);
        rcc = regionprops(label == n, 'Solidity');  % convex area
        rc(n) = cat(1, rcc.Solidity);
        ll = regionprops(label == n, 'MajorAxisLength');  % length
        l(n) = cat(1, ll.MajorAxisLength);
        ww = regionprops(label == n, 'MinorAxisLength');  % width
        w(n) = cat(1, ww.MinorAxisLength);
    end
    rl = l ./ w;    % ratio between length and width 
    
    % find the component which need to be segmented
    k1 = find(a > Ra);
    k2 = find(rc < Rc);
    k3 = find(rl > Rl);
    k = [k1; k2; k3];
    k = unique(k);
    
    if length(k) ~= 0
        
    % initial contour
    bw2 = zeros(s1, s2);
    for m = 1 : length(k)
        pp = find(label == k(m));
        bw2(pp) = 1;
    end
    bw2 = bwareaopen(bw2, Ra_min);
    img_Dist = bwdist(~bw2,'cityblock');   % distance transform
    imgDist = -img_Dist;
    imgDist(~bw2)=-inf;
    Dis_img = imregionalmin(imgDist);
    dis = Dis_img.*bw2;      % local minuma
    dis = imdilate(dis, se);
        
    [label1, num1] = bwlabel(dis, 8);
    t = 0:0.05:6.28;
    for n1 = 1 : num1
        cen = regionprops(label1 == n1, 'centroid');   % center
        cen = cat(1, cen.Centroid);   % cen(:,1): horizontal, cen(:,2): vertical
        r = img_Dist(round(cen(2)), round(cen(1))) / sqrt(2);  % radium of initial circle
        if r == 0
            r = 2;
        end
        x = double(cen(1) + r * cos(t));
        y = double(cen(2) + r * sin(t));
        [x, y] = snakeinterp(x, y, Dmax, Dmin);
        [x, y] = polybool('intersection', s_2, s_1, x, y);
                             % in case polygon vertex is outside of image
                             % x: horizontal, y: vertical
         x = x';
         y = y';
            
         % snake deformation
         for i=1 : ceil(iter/5)
             if i <= floor(iter/5) 
                 [x, y] = snakedeform(x, y, alpha, beta, gamma, kappa, px, py, 5);
             else
                 [x, y] = snakedeform(x, y, alpha, beta, gamma, kappa, px, py, iter-floor(iter/5)*5);
             end
             [x, y] = snakeinterp(x, y, Dmax, Dmin);
         end  
         xx = ceil(x); yy = ceil(y);
         len = length(xx);
         for i = 1:len
             if xx(i) <= s2 & yy(i) <= s1
                bw1(yy(i),xx(i)) = 0;
             end
         end
                
    end
               
    else
        break;
    end

end



end

