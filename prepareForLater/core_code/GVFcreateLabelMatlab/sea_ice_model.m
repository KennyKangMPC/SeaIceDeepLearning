function [floe, brash] = sea_ice_model( ice_floe, brash_ice, img )
% Sea-ice model. Polygon-fit for each sea-ice floe,
%                and disk-fit for each brash ice.
% 
% Input data:
% ice_floe: structure data of identified sea-ice floes, 
%           includes: Vertices, Center, Area, Perimeter
% brash_ice: structure data of identified brash ice, 
%           includes: Vertices, Center, Area, Perimeter
% img:   segmented binary image
% 
% Output data:
% floe:      structure data, includes: Vertices, Center, Area, 
%            Perimeter, and overlapping floes 
% brash:     structure data, includes: Vertices, Center, Area, 
%            Perimeter, and overlapping brash 
%
%    Copyright:     NTNU, Department of Marine Technology
%    Project:	    Arctic DP, RCN no.199567
%    Author:        Qin Zhang
%    Date created:  2015.08.07  Qin Zhang
%    Revised:      	2015.08.29  Qin Zhang

%%
t = 0:0.05:6.28;

%% sea-ice floe model
floe = [];
bw_floe = zeros(size(img));

% floe polygonization
ver = [];
for i = 1 : length(ice_floe)
    pixels = cat(1, ice_floe(i).PixelsPosition);
    x = pixels(:, 1);
    y = pixels(:, 2);
    k = convhull(x, y, 'simplify', true);
    v = [x(k), y(k)]; 
    ver0 = struct('Vertices', v);
    ver = [ver; ver0];           
end 
     
for i = 1 : length(ver)
    v = cat(1, ver(i).Vertices);
    
    bw = roipoly(double(img), v(:, 1), v(:, 2));  % region of interest
    bw_floe(bw == 1) = 1;    % floe model
    
%     c = regionprops(bw == 1, 'centroid');   % center
%     c = cat(1, c.Centroid);   
%     a = polyarea(v(:,1), v(:,2));   % area
%     p = regionprops(bw == 1, 'perimeter');   % perimeter
%     p = cat(1, p.Perimeter); 
    [geom, iner, cpmo] = polygeom(v(:,1), v(:,2));
    a = geom(1);    % area
    c = [geom(2), geom(3)];    % center
    p = geom(4);    % perimeter
    
    % floe-floe overlapping 
    inter_floe = [];
    for j = 1 : length(ver)
        if j ~= i
            v0 = cat(1, ver(j).Vertices);
            [xx, yy] = polyxpoly(v(:,1), v(:,2), v0(:,1), v0(:,2));
            if xx ~= NaN
                inter_floe = [inter_floe; j];
            end
        end
    end
       
    % floe-brash overlapping 
    inter_brash = [];
    for k = 1 : length(brash_ice)
        cc = cat(1, brash_ice(k).Center);   % center
        aa = cat(1, brash_ice(k).Area);   % area
        r = sqrt(aa / pi);   % radius
        x = cc(1) + r * cos(t);
        y = cc(2) + r * sin(t);
    
        [xx, yy] = polyxpoly(v(:,1), v(:,2), x, y);
        if xx ~= NaN
            inter_brash = [inter_brash; k];
        end
        
    end   
    
    intersect = struct('floe', inter_floe, 'brash', inter_brash);
   
    floe0 = struct('Vertices', v, 'Center', c, 'Area', a, 'Perimeter', p, 'Intersect', intersect);
    floe = [floe; floe0];           

end 

figure, imshow(bw_floe), hold on,  
for i = 1 : length(floe)
    v = cat(1, floe(i).Vertices);
    c = cat(1, floe(i).Center);
    line(v(:, 1), v(:, 2));
    plot(c(1), c(2), 'r+');
end
hold off


%% brash ice model
brash = [];
for i = 1 : length(brash_ice)
    c = cat(1, brash_ice(i).Center);   % center
    a = cat(1, brash_ice(i).Area);   % area
    r = sqrt(a / pi);   % radius
    p = 2 * pi * r;   % perimeter
    
    x = c(1) + r * cos(t);    % circle
    y = c(2) + r * sin(t);
     
    % brash-brash overlapping 
    inter_brash = [];   
    for j = 1 : length(brash_ice)
        if j ~= i
            c0 = cat(1, brash_ice(j).Center);
            a0 = cat(1, brash_ice(j).Area);   % area
            r0 = sqrt(a0 / pi);   % radius
            d = sqrt( (c(1)-c0(1))^2 + (c(2)-c0(2))^2 );
            rr = r + r0;
            if d < rr
                inter_brash = [inter_brash; j];
            end
        end
    end
    
    % brash-floe overlapping
    inter_floe = [];
    for k = 1 : length(floe)
        v = cat(1, floe(k).Vertices);
        [xx, yy] = polyxpoly(v(:,1), v(:,2), x, y);
        if xx ~= NaN
            inter_floe = [inter_floe; k];
        end
        
    end   
    
    intersect = struct('floe', inter_floe, 'brash', inter_brash);
    
    brash0 = struct('Radius', r, 'Center', c, 'Area', a, 'Perimeter', p, 'Intersect', intersect);
    brash = [brash; brash0];
    
end 

bw_brash = zeros(size(img));
for i = 1 : length(brash)
    c = cat(1, brash(i).Center);
    r = cat(1, brash(i).Radius);
    x = c(1) + r * cos(t);
    y = c(2) + r * sin(t);
    
    bw = roipoly(double(img), x, y);  % region of interested
    bw_brash(bw == 1) = 1;
end

figure, imshow(bw_brash), hold on, 
for i = 1 : length(brash)
    c = cat(1, brash(i).Center);
    r = cat(1, brash(i).Radius);
    x = c(1) + r * cos(t);
    y = c(2) + r * sin(t);  
    plot(c(1), c(2), 'r.');
    plot(x, y);
end
hold off
        

