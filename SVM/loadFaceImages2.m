function [images labels] = loadFaceImages2(filename,sampling, preprocess)

if nargin<2
    sampling =1;
end

if nargin < 3
    preprocess = 0;
end

% this is a flag that allow you to activate/deactivate the data augmentation
% Data augmentation will increase the size of the dataset by created variations 
%(mirroring, flipping, displacements) of each given image. This aims to produce more
% training images and, therefore, improve performance
augmented=1;


fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);


line1=fgetl(fp);
line2=fgetl(fp);

numberOfImages = fscanf(fp,'%d',1);

images=[];
labels =[];
for im=1:sampling:numberOfImages
    
    label = fscanf(fp,'%d',1);
    
    labels= [labels; label];
    
    imfile = fscanf(fp,'%s',1);
    I=imread(imfile);
    if size(I,3)>1
        I=rgb2gray(I);
    end
    %Pre-process
    if preprocess
        I = histeq(I);
    end

    vector = reshape(I,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;

    images= [images; vector];
    
    if augmented
        % For both face and non-face images, apply the same 4 augmentations:
            
            % 1. Horizontal flip
            Itemp = fliplr(I);
            vector = reshape(Itemp, 1, size(I, 1) * size(I, 2));
            vector = double(vector);
            images = [images; vector];
            labels = [labels; label];
            
            % 2. Shift right by 1 pixel
            Itemp = circshift(I, [0 1]);
            vector = reshape(Itemp, 1, size(I, 1) * size(I, 2));
            vector = double(vector);
            images = [images; vector];
            labels = [labels; label];
            
            % 3. Shift left by 1 pixel
            Itemp = circshift(I, [0 -1]);
            vector = reshape(Itemp, 1, size(I, 1) * size(I, 2));
            vector = double(vector);
            images = [images; vector];
            labels = [labels; label];
            
            % 4. Horizontal flip + shift right
            Itemp = circshift(fliplr(I), [0 1]);
            vector = reshape(Itemp, 1, size(I, 1) * size(I, 2));
            vector = double(vector);
            images = [images; vector];
            labels = [labels; label];

            % 5. Vertical flip
            Itemp =flipud(I);
            vector = reshape(Itemp,1, size(I, 1) * size(I, 2));
            vector = double(vector); % / 255;
            images= [images; vector];
            labels= [labels; label];   

    end
    
end

fclose(fp);

end
