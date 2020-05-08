% This part of the code is developed based on algorithm for sea-ice image
% processing given by Qin Zhang and Roger Skjetne. The algorithm is called
% GVF snake operates given in their paper "Image Processing for 
% Identification of Sea-Ice Floes and the Floe Size Distributions"

%% Preparing work
warning('off')
clear
close all
clc
disp('start processing')
%% parameter setting
% parameters for image
kms0 = 3;    % kmeans

Ra_min = 10;  % minimum ice piece area
Ra = 2500;   % maximum ice piece area
Rc = 0.9;   % convexity threshlod
Rl = 2;    % threshold ratio between length and width

se = strel('disk', 3);  % morphology structure element

% parameters for GVF Snake
Num = 500;      % number of GVF iterations
iter = 100;   % number of Snake iterations 
% N = 50;    % iter of deformation

% parameters for shape enhancement
se_th = 50;    % threshold for adaptive morphology structure element
min_floe = 40;   % minimum floe area, default is 40
min_brash = 1;   % minimum brash area


%% other parameters for GVF Snake, DON'T TUNE!
sigma = 0;   % gaussianBlur
GradientOn = 1;  % 1 : Gradient on  0 : Gradient off

GVFOn = 1;   % 1 : GVF  0 : SVF
mu = 0.1;    % parameter for GVF (more noise, increase mu)

alpha = 0.05;   % internal weight that control snake's tension
beta = 0;     % internal weight that control snake's rigidity
gamma = 1;    % step size in one iteration
kappa = 0.5;   % external force weight
Dmin = 0;   % min raster of the snake
Dmax = 1;   % max raster of the snake

timer = 1;

%% Here work on creating labels via for loops since
% We need to read files from folder via a loop and then create those labels

% for training
files = dir('../label_create/training/images');
mkdir('../label_create/training/labels')
for i = 1:numel(files)-2
    I = imread(['../label_create/training/images/',files(i+2).name]);
    [seg, bk] = seaice_kmean_GVF_forenhancement( I, kms0, sigma, GradientOn, GVFOn, Num, mu,...
        iter, alpha, beta, gamma, kappa, Dmin, Dmax,  Ra_min, Ra, Rc, Rl, se, timer);
    
    [out, index_floe, ice_floe, index_brash, brash_ice, index_slush, index_water, index_residue, coverage, imMask] = ...
        ice_shape_enhancement(bk, seg, min_floe, min_brash, se_th);
    
    fn = files(i+2).name;
    imwrite(im2bw(im2double(imMask),0.99), ['../label_create/training/labels/', fn])
    disp(['finish training #',num2str(i),' image labeling'])
end

% for testing
files = dir('../label_create/test/images');
mkdir('../label_create/test/labels')
for i = 1:numel(files)-2
    I = imread(['../label_create/test/images/',files(i+2).name]);
    [seg, bk] = seaice_kmean_GVF_forenhancement( I, kms0, sigma, GradientOn, GVFOn, Num, mu,...
        iter, alpha, beta, gamma, kappa, Dmin, Dmax,  Ra_min, Ra, Rc, Rl, se, timer);
    
    [out, index_floe, ice_floe, index_brash, brash_ice, index_slush, index_water, index_residue, coverage, imMask] = ...
        ice_shape_enhancement(bk, seg, min_floe, min_brash, se_th);
    
    fn = files(i+2).name;
    imwrite(im2bw(im2double(imMask),0.99), ['../label_create/test/labels/', fn])
    disp(['finish testing #',num2str(i),' image labeling'])
end
