
%pic = imread('test.jpg');
pic = imread('icefloe.png'); 
ima = rgb2gray(pic);
figure, imshow(ima);

k = 3;

% check image
ima = double(ima);
copy = ima;         % make a copy
ima = ima(:);       % vectorize ima
mi = min(ima);      % deal with negative 
ima = ima - mi + 1;     % and zero values

s = length(ima);

% create image histogram

m = max(ima) + 1;
h = zeros(1, m);
hc = zeros(1, m);
for i = 1 : s
  if(ima(i) > 0)
      h(ima(i)) = h(ima(i)) + 1;
  end
end
ind = find(h);
hl = length(ind);

% initiate centroids
  
mu = (1:k) * m / (k+1);

% start process

while(true)
  
  oldmu = mu;
  % calculate the distance to classify  
 
  for i = 1 : hl
      c = abs(ind(i) - mu);
      cc = find(c == min(c));
      hc(ind(i)) = cc(1);
  end
  
  % calculate new means to find out the new centroids
  
  for i = 1 : k, 
      a = find(hc == i);
      mu(i) = sum(a .* h(a)) / sum(h(a));
  end
  
  if(mu == oldmu)
      break;
  end;
  
end

% calculate mask
s = size(copy);
mask = zeros(s);
mask1 = mask;
for i = 1 : s(1),
    for j = 1 : s(2),
        c = abs(copy(i, j) - mu);
        a = find(c == min(c));  
        mask(i,j) = a(1);
    end
end

 for i = 1 : k
	p = find(mask == i);
	mask1(p) = 1 / k * i;
    
    for j = 1 : length(p)
        ss(j) = copy(p(j));
    end
    average_intensity(i) = sum(ss(:)) / length(p);
    
    n(i) = sum(sum(mask == i));
end

% for i = 1 : k
%     n(i) = sum(sum(mask == i));
% end
% [r,c]=size(mask);
ic = n / (s(1) * s(2));

% [B,L] = bwboundaries(BW);
figure, imshow(mask1);
% colormap('default');
% hold on;
% for k = 1:length(B) 
%   boundary = B{k}; 
%   plot(boundary(:,2), boundary(:,1), 'k', 'LineWidth', 2);
% end 
% hold on;
% axis off;

