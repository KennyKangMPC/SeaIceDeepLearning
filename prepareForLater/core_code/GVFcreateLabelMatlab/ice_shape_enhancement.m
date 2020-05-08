function [out, index_floe, ice_floe, index_brash, brash_ice, index_slush, ...
    index_water, index_residue, coverage, rgb] = ice_shape_enhancement(bk, seg, min_floe, min_brash, se_th)
% sea ice shape enhancement
% 
% Input data:
% bk:              binary ice image by kmeans method
% seg:             sea ice segmentation image, the output of function
%                  seaice_kmean_GVF
% min_floe:        threshold of minimum size of ice floe
% min_brash:       threshold of minimum size of brash ice
% se_th:           threshold of element structure for morphology operator
%
% Output data:
% out:             morphology cleaned ice pieces (ice floe & brash ice)
% index_floe:      layer of ice floe 
% ice_floe:        ice floe information (areas, centers, perimeter, and
%                                        pixel positions)
% index_brash:     layer of brash ice 
% brash_ice:       brash ice information (areas, centers, perimeter, and
%                                         pixel positions)
% index_slush:     layer of slush ice 
% index_water:     layer of water 
% index_residue:   layer of residue
% coverage:        percentage of floe, brash, slush and water


bw = seg == 1;      % light ice segmentation
bw = double(bw);
k = seg == 0.5;      % dark ice segmenation
k = double(k);

l = zeros(size(bw));   % labeled sea ice image
ice_area = [];
[label_bw, nn_bw] = bwlabel(bw, 4);
for i = 1 : nn_bw
    p = find(label_bw == i);
    l(p) = i;
    area0 = length(p);
    ice_area = [ice_area, area0];
end

k = k - k.*bw;
[label_k, nn_k] = bwlabel(k, 4);
for i = 1 : nn_k
    p = find(label_k == i);
    l(p) = i + nn_bw;
    area0 = length(p);
    ice_area = [ice_area, area0];
end

[A, ind] = sort(ice_area);

out =  zeros(size(bw));    % filled hole, morphology, labeled sea ice image
fill = zeros(size(bw));    % filled hole, labeled sea ice image
t = 0;
for i = 1 : max(max(l))
    p = find(l == ind(i));
    b = zeros(size(out));
    b(p) = 1;
    
    b = imfill(b,  'hole');   % fill hole
    
    [ll, kk] = bwlabel(b, 4);
    for j = 1 : kk
        pp = find(ll == j);
        if length(pp) > 0
            fill(pp) = 1;
        end
    end
    
    r = length(p);
    if r < se_th
        r = 1;
    else
        r = 2;
    end
    se = strel('disk', r);
    
    b = imclose(b, se);   % morphology
    b = imopen(b, se);
    b = imfill(b,  'hole');   % fill hole
        
    [ll, kk] = bwlabel(b, 4);
    for j = 1 : kk
        pp = find(ll == j);
        if length(pp) > 0
            t = t + 1;
            out(pp) = t;
        end
    end
end


%% floe + brash
ice_area = [];       % ice area
floe_area = [];           % floe area
floe_cen = [];        % floe center
color_floe = [];     
brash_area = [];          % brash area
brash_cen = [];      % brash center
color_brash = [];
index_floe = zeros(size(out));      % floe map
index_brash = zeros(size(out));     % brash map
ice_floe = [];          % structure of ice floe information
brash_ice = [];         % sturcture of brash ice information

for i = 1: max(max(out))
    p = find(out == i);
    area0 = length(p);   % area
    [r, c] = find(out == i);   % pixel position
    pixels = [c, r];
    
    if area0 ~= 0
        ice_area = [ice_area, area0];
        color_label = fix( (1 - exp(-area0/1000)) * 10000 );
        
        cen = regionprops(out == i, 'centroid');   % center
        cen = cat(1, cen.Centroid);
        
        per = regionprops(out == i, 'perimeter');   % perimeter
        per = cat(1, per.Perimeter);
        
        s0 = struct('Center', cen, 'Area', area0, 'Perimeter', per,...
            'PixelsPosition', pixels);
        
        if area0 > min_floe                % ice floe
            index_floe(p) = color_label;
            floe_area = [floe_area, area0];
            color_floe = [color_floe, color_label];
            floe_cen = [floe_cen; cen];
            ice_floe = [ice_floe; s0];
            
        elseif area0 > min_brash             % brash ice
            index_brash(p) = color_label;
            brash_area = [brash_area, area0];
            color_brash = [color_brash, color_label];
            brash_cen = [brash_cen; cen];
            brash_ice = [brash_ice; s0];
        end
    end
end

index = index_floe + index_brash;   % floe + brash
p = find(index ~= 0);

%% slush + water
bk0 = bk;
bk0(p) = 1;

index_slush = bk0;
index_slush(p) = 0;      % slush map

index_water = 1 - bk0;    % water map

index_residue =  ones(size(fill));
index_residue(find(fill ~= 0)) = 0;
index_residue = index_residue.*index_slush;    % residue map

%% percentage
floe = sum(floe_area) / (size(out, 1)*size(out, 2));
brash = sum(brash_area) / (size(out, 1)*size(out, 2));

sp = find(index_slush == 1);
slush = length(sp) / (size(out, 1)*size(out, 2));

wp = find(index_water == 1);
water = length(wp) / (size(out, 1)*size(out, 2));

coverage = struct('IceFloe', floe, 'BrashIce', brash, 'Slush', slush, 'Water', water);
%% rgb
rgb = label2rgb(index, @jet, [1,1,1]);
%figure,imshow(rgb);


% hold on
% for i = 1 : length(floe_cen)
%     plot(floe_cen(i, 1), floe_cen(i, 2), 'k*')
% end
% for i = 1 : length(brash_cen)
%     plot(brash_cen(i, 1), brash_cen(i, 2), 'k.')
% end
% axis off
% 
% area_ice = [color_floe, color_brash];
% colormap(jet);
% n = 6;
% d = fix((max(area_ice)-min(area_ice))/n);
% ysh = min(area_ice) : d : max(area_ice);
% hcb = colorbar;
% ytic = get(colorbar, 'Ytick'); 
% set(colorbar, 'YTick', linspace(min(ytic), max(ytic), length(ysh)));
% YT = [];
% for i = 1 : length(ysh)
%     YT{1, i} = -round(1000 * log(1 - ysh(i)/10000));
% end
% set(colorbar, 'YTickLabel', YT)


%% histogram
% Since we just want the segmentation result, therefore, this part of the
% codes can be ignored here. Below is only for the histogram, or so
% called distribution of the size of the sea ice code.

% figure,
% nbins = 50;
% [z, n] = hist(floe_area, nbins);
% h = bar(n(1 : nbins), z(1 : nbins));
% ch = get(h,'Children');
% fvd = get(ch,'Faces');
% fvcd = get(ch,'FaceVertexCData');
% [zs, izs] = sortrows(z', 1);
% k = 255;
% colormap(jet(k));
% for i = 1 : nbins
%     color(i) = fix( (1 - exp(-n(i)/1000)) * 10000 );
%     fvcd(fvd(i,:)) = color(i);
% end
% set(ch,'FaceVertexCData',fvcd)
% 
% colormap(jet);
% nn = 8;
% d = fix((max(color_floe)-min(color_floe))/nn);
% ysh = min(color_floe) : d : max(color_floe);
% hcb = colorbar;
% ytic = get(colorbar, 'Ytick'); 
% set(colorbar, 'YTick', linspace(min(ytic), max(ytic), length(ysh)));
% YT = [];
% for i = 1 : length(ysh)
%     YT{1, i} = -round(1000 * log(1 - ysh(i)/10000));
% end
% set(colorbar, 'YTickLabel', YT)

end













