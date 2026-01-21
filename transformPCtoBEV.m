function transformPCtoBEV(lidarData,gridParams,dataLocation)
% createBEVData create the Bird's-Eye-View image data adn the corresponding
% labels from the given dataset.
%
% Copyright 2021 The MathWorks, Inc.

% Reset the point cloud datastore.
reset(lidarData);
i=1;
while hasdata(lidarData)

    ptCloud = read(lidarData);
    [processedData,~] = preprocess(ptCloud,gridParams);

    if ~isfolder(dataLocation)
        mkdir(dataLocation);
    end

    imgSavePath = fullfile(dataLocation,sprintf('%04d.jpg',i));
    imwrite(processedData,imgSavePath);
    i = i + 1;
end
end

% Get the BEV image from point cloud.
function [imageMap,ptCldOut] = preprocess(ptCld,gridParams)

    pcRange = [gridParams{1,1}{1} gridParams{1,1}{2} gridParams{1,1}{3} ...
               gridParams{1,1}{4} gridParams{1,1}{5} gridParams{1,1}{6}]; 

    indices = findPointsInROI(ptCld,pcRange);
    ptCldOut = select(ptCld,indices);
    
    bevHeight = gridParams{1,2}{2};
    bevWidth = gridParams{1,2}{1};
    
    % Find grid resolution.
    gridH = gridParams{1,3}{2};
    gridW = gridParams{1,3}{1};
    
    loc = ptCldOut.Location;

    hasIntensity = false;
    if isfield(ptCldOut,'Intensity')
        hasIntensity = true;
        intensity = ptCldOut.Intensity;
        intensity = normalize(intensity,'range');
    end
    
    % Find the grid each point falls into.
    loc(:,1) = int32(floor(loc(:,1)/gridH)+bevHeight/2) + 1;
    loc(:,2) = int32(floor(loc(:,2)/gridW)+bevWidth/2) + 1;
    
    % Normalize the height.
    loc(:,3) = loc(:,3) - min(loc(:,3));
    loc(:,3) = loc(:,3)/(pcRange(6) - pcRange(5));
    
    % Sort the points based on height.
    [~,I] = sortrows(loc,[1,2,-3]);
    locMod = loc(I,:);

    if hasIntensity
        intensityMod = intensity(I,:);
    end
    
    % Initialize height and intensity map
    heightMap = zeros(bevHeight,bevWidth);
    if hasIntensity
        intensityMap = zeros(bevHeight,bevWidth);
    end
    
    locMod(:,1) = min(locMod(:,1),bevHeight);
    locMod(:,2) = min(locMod(:,2),bevWidth);
    
    % Find the unique indices having max height.
    mapIndices = sub2ind([bevHeight,bevWidth],locMod(:,1),locMod(:,2));
    [~,idx] = unique(mapIndices,"rows","first");
    
    binc = 1:bevWidth*bevHeight;
    counts = hist(mapIndices,binc);
    
    normalizedCounts = min(1.0, log(counts + 1) / log(64));
    
    for i = 1:size(idx,1)
        heightMap(mapIndices(idx(i))) = locMod(idx(i),3);
        if hasIntensity
            intensityMap(mapIndices(idx(i))) = intensityMod(idx(i),1);
        end
    end
    
    densityMap = reshape(normalizedCounts,[bevHeight,bevWidth]);
    
    imageMap = zeros(bevHeight,bevWidth,3);
    imageMap(:,:,1) = densityMap;       % R channel
    imageMap(:,:,2) = heightMap;        % G channel
    if hasIntensity
        imageMap(:,:,3) = intensityMap;     % B channel
    end
end