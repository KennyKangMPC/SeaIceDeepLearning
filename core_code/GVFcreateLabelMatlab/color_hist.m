
floe_area = cat(1, floe.Area);
color_floe = fix( (1 - exp(-floe_area/1000)) * 10000 );

min_x = 20;
inter = 70;
max_x = 3500;

color_min = fix( (1 - exp(-min_x/1000)) * 10000 );
color_max = fix( (1 - exp(-max_x/1000)) * 10000 );


%% histogram
figure,

% nbins = 50;
% [z, n] = hist(floe_area, nbins);
[z, n] = hist(floe_area, min_x : inter : max_x);
nbins = length(z);
n = n + inter/2;

h = bar(n(1 : nbins), z(1 : nbins));
ch = get(h,'Children');
fvd = get(ch,'Faces');
fvcd = get(ch,'FaceVertexCData');
[zs, izs] = sortrows(z', 1);
k = 255;
colormap(jet(k));
for i = 1 : nbins
    color(i) = fix( (1 - exp(-n(i)/1000)) * 10000 );
    fvcd(fvd(i,:)) = color(i);
end
set(ch,'FaceVertexCData',fvcd)

colormap(jet);
nn = 8;
d = fix((color_max-color_min)/nn);
ysh = min(color_min) : d : max(color_max);
hcb = colorbar;
ytic = get(colorbar, 'Ytick'); 
set(colorbar, 'YTick', linspace(min(ytic), max(ytic), length(ysh)));
YT = [];
for i = 1 : length(ysh)
    YT{1, i} = -round(1000 * log(1 - ysh(i)/10000));
end
set(colorbar, 'YTickLabel', YT)
