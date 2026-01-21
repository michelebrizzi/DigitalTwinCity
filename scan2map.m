function ptCloudMap = scan2map(imgList, pcList, bevList, poses, cameraParams, R_CL, t_CL)

% Inizializzazioni empty viewset
vSet = pcviewset;

pcProcessed = cell(1,length(pcList));
lidarScans2d = cell(1,length(pcList)); 
submaps = cell(1,length(pcList)/nScansPerSubmap);
pcsToView = cell(1,length(pcList)); 

% Create a loop closure detector
loopDetector = scanContextLoopDetector;

% Carica detector voxel R-CNN pre-addestrato
useDetector = false;
if useDetector
    % detector = voxelRCNNObjectDetector("kitti");
    % detector = load("models/complex-yolov4/cspdarknet-complex-yolov4.mat").detector;
    detector = load('pretrainedPointPillarsDetector.mat','detector').detector;
end

% Initialize transformations
absTform = rigidtform3d;  % Absolute transformation to reference frame
relTform = rigidtform3d;  % Relative transformation between successive scans

% Parametri
maxLidarRange = 70;
gridDown     = 0.2;      % m, downsample per registrazione
planeDist    = 0.15;      % m, soglia RANSAC per piano
planeNormal  = [0 0 1];   % normale attesa del suolo
planeTiltDeg = 7;         % tolleranza inclinazione
regGridSize  = 1.0;       % m, pcregisterndt
maxRMSELoop  = 3.0;       % m, soglia accettazione loop
mergeGrid    = 0.05;      % m, voxel per pcmerge finale
xMin = -500.0;     
xMax = 500.0;      
yMin = -500.0;      
yMax = 500.0;      
zMin = -70.0;     
zMax = 150.0;  
% bevHeight = 608;
% bevWidth = 608;
% gridW = (yMax - yMin)/bevWidth;
% gridH = (xMax - xMin)/bevHeight;
% gridParams = {{xMin,xMax,yMin,yMax,zMin,zMax},{bevWidth,bevHeight},{gridW,gridH}};
distanceMovedThreshold = 0.3;

traj = makeTrajectory(poses,R_CL,t_CL); % struct con interpPos(t), interpQuat(t)

if viewPC==1
    pplayer = pcplayer([xMin xMax],[yMin yMax],[zMin zMax],'MarkerSize',10);
end

% If you want to view the created map and posegraph during build process
if viewMap==1
    ax = newplot; % Figure axis handle
    view(20,50);
    grid on;
end

if viewPose==1
    patch = poseplot;
end

% mappa iniziale = prima scan
viewId = 1;
for k = 1:numel(pcList)

    fprintf("Elaborando step %d...\n", k);

    % Read images
    Ik = imread(imgList(k));

    % Read point cloud
    [pcK, tK] = readPointCloudWithTimestamp(pcList(k));
    pcK = colorizePointCloud(pcK, Ik, cameraParams, R_CL, t_CL);

    % Read BEV
    bevk = imread(bevList(k));

    % Read poses
    [Rk_gt, Tk_gt] = poseAtTime(traj, tK);

    Twk_gt = eye(4); Twk_gt(1:3,1:3)=Rk_gt; Twk_gt(1:3,4)=Tk_gt(:);
    absTform = rigidtform3d(Twk_gt);

    % Process point cloud
    %   - Segment and remove ground plane
    %   - Segment and remove ego vehicle
    %   - Downsample point cloud
    pcNG = processPointCloud(pcK,gridDown);

    if viewPC==1
        % Visualize down sampled point cloud
        view(pplayer, pcNG);
        pause(0.001)
    end

    firstFrame = (k==1);
    if firstFrame
        vSet = addView(vSet, viewId, absTform, "PointCloud", pcK);

        % Extract the scan context descriptor from the first point cloud
        descriptor = scanContextDescriptor(pcK);

        % Add the first descriptor to the loop closure detector
        addDescriptor(loopDetector, viewId, descriptor);

        viewId = viewId + 1;
        ptCloudPrev = pcNG;

        prevAbsTform = absTform;  % Absolute transformation to reference frame
        continue;
    end

    % stima relativa con prior: NDT (robusto a outlier sparsi)
    initTform = rigidtform3d(prevAbsTform.A \ absTform.A); 
    [relTform, ~] = pcregisterndt(pcNG, ptCloudPrev, regGridSize, "InitialTransform", initTform);

    % ptCloudOut = undistortEgoMotion(pcNG,relTform,pointTimestamps,sweepTime);

    relPose = [tform2trvec(relTform.T') tform2quat(relTform.T')];

    if sqrt(norm(relPose(1:3))) > distanceMovedThreshold
        addRelativePose(pGraph,relPose);
        scanAccepted = 1;
    else
        scanAccepted = 0;
    end

    if scanAccepted == 1
        count = count + 1;
        
        pcProcessed{count} = pcl_wogrd_sampled;
        
        lidarScans2d{count} = exampleHelperCreate2DScan(pcl_wogrd_sampled);
        
        % Submaps are created for faster loop closure query. 
        if rem(count,nScansPerSubmap)==0
            submaps{count/nScansPerSubmap} = exampleHelperCreateSubmap(lidarScans2d,...
                pGraph,count,nScansPerSubmap,maxLidarRange);
        end
        
        % loopSubmapIds contains matching submap ids if any otherwise empty.   
        if (floor(count/nScansPerSubmap)>subMapThresh)
            [loopSubmapIds,~] = exampleHelperEstimateLoopCandidates(pGraph,...
                count,submaps,lidarScans2d{count},nScansPerSubmap,...
                loopClosureSearchRadius,loopClosureThreshold,subMapThresh);
            
            if ~isempty(loopSubmapIds)
                rmseMin = inf;
                
                % Estimate best match to the current scan
                for k = 1:length(loopSubmapIds)
                    % For every scan within the submap
                    for j = 1:nScansPerSubmap
                        probableLoopCandidate = ...
                            loopSubmapIds(k)*nScansPerSubmap - j + 1;
                        [loopTform,~,rmse] = pcregisterndt(pcl_wogrd_sampled,...
                            pcProcessed{probableLoopCandidate},gridStep);
                        % Update best Loop Closure Candidate
                        if rmse < rmseMin
                            loopCandidate = probableLoopCandidate;
                            rmseMin = rmse;
                        end
                        if rmseMin < rmseThreshold
                            break;
                        end
                    end
                end
                
                % Check if loop candidate is valid
                if rmseMin < rmseThreshold
                    % loop closure constraint
                    relPose = [tform2trvec(loopTform.T') tform2quat(loopTform.T')];
                    
                    addRelativePose(pGraph,relPose,infoMat,...
                        loopCandidate,count);
                    numLoopClosuresSinceLastOptimization = numLoopClosuresSinceLastOptimization + 1;
                end
                     
            end
        end
        if (numLoopClosuresSinceLastOptimization == optimizationInterval)||...
                ((numLoopClosuresSinceLastOptimization>0)&&(i==length(pClouds)))
            if loopClosureSearchRadius ~=1
                disp('Doing Pose Graph Optimization to reduce drift.');
            end
            % pose graph optimization
            pGraph = optimizePoseGraph(pGraph);
            loopClosureSearchRadius = 1;
            if viewMap == 1
                position = pGraph.nodes;
                % Rebuild map after pose graph optimization
                omap = occupancyMap3D(mapResolution);
                for n = 1:(pGraph.NumNodes-1)
                    insertPointCloud(omap,position(n,:),pcsToView{n}.removeInvalidPoints,maxLidarRange);
                end
                mapUpdated = true;
                ax = newplot;
                grid on;
            end
            numLoopClosuresSinceLastOptimization = 0;
            
            % Reduce the frequency of optimization after optimizing the trajectory
            optimizationInterval = optimizationInterval*7;
        end
    end


    % Update absolute transformation to reference frame (first point cloud): 
    % T_w^{(k)} = T_w^{(k-1)} * T_{k-1}^{k} 
    % absTform = rigidtform3d(absTform.A * relTform.A);
    prevAbsTform = absTform;

    % Add current point cloud scan as a view to the view set
    vSet = addView(vSet, viewId, absTform, "PointCloud", pcK);

    % Add a connection from the previous view to the current view representing the relative
    % transformation between them
    vSet = addConnection(vSet,viewId-1,viewId,relTform);

    % Extract the scan context descriptor from the point cloud
    descriptor = scanContextDescriptor(pcK);

    % Add the descriptor to the loop closure detector
    addDescriptor(loopDetector,viewId,descriptor)

    % Detect loop closure candidates
    loopViewId = detectLoop(loopDetector);

    % A loop candidate was found
    if ~isempty(loopViewId)
        loopViewId = loopViewId(1);
        
        % Retrieve point cloud from view set
        loopView = findView(vSet, loopViewId);
        ptCloudOrig = loopView.PointCloud;
        
        % Process point cloud
        ptCloudOld = processPointCloud(ptCloudOrig,gridDown);
        
        % Downsample point cloud
        ptCloudOld = pcdownsample(ptCloudOld,"gridAverage", gridDown);
        
        % Use registration to estimate the relative pose
        [relTform,~,rmse] = pcregisterndt(pcNG, ptCloudOld, regGridSize,"MaxIterations",50);
        
        acceptLoopClosure = rmse <= maxRMSELoop;
        if acceptLoopClosure
            % For simplicity, use a constant, small information matrix for
            % loop closure edges
            infoMat = 0.01 * eye(6);
            
            % Add a connection corresponding to a loop closure
            vSet = addConnection(vSet, loopViewId, viewId, relTform, infoMat);
        end
    end

    % detection 3D su non-ground e masking dinamici
    if useDetector
        try
            % [bboxes3d, scores, labels] = detect(detector, pcNG, "MinScore", 0.4); %voxelnetrcnn
            % [bboxes, scores, labels] = detect(detector, bevk); % complex-yolov4
            % [ptCldOut,bboxCuboid] = transferbboxToPointCloud(bboxes,gridParams,pcK);
            [bboxes,score,labels] = detect(detector,pcNG); idx = score>.65;
            % helperDisplayBoxes(pcK,bboxes(idx,:),labels(idx));
            pcNG = removePointsInsideBBoxes(pcNG, bboxes(idx,:));
        catch
            % prosegui senza detection
        end
    end

    % prepara per iterazione successiva
    ptCloudPrev = pcNG;
    viewId = viewId + 1;

    if viewMap==1
        % Insert point cloud to the occupancy map in the right position
        position = pGraph.nodes(count);
        insertPointCloud(omap,position,pcToView.removeInvalidPoints,maxLidarRange);

        if (rem(count-1,15)==0)||mapUpdated
            exampleHelperVisualizeMapAndPoseGraph(omap, pGraph, ax);
        end
        mapUpdated = false;
    else
        % Give feedback to know that example is running
        if (rem(count-1,15)==0)
            fprintf('.');
        end
    end

    if viewPose
        set(patch,Orientation=Rk_gt,Position=Tk_gt);
    end
    drawnow


end

% Ottimizzazione pose graph e costruzione mappa finale ===
G = createPoseGraph(vSet);
optimG = optimizePoseGraph(G, 'g2o-levenberg-marquardt');
vSetOpt = updateView(vSet, optimG.Nodes);

ptcs  = vSetOpt.Views.PointCloud;
poses = vSetOpt.Views.AbsolutePose;     % rigidtform3d assoluti ottimizzati
mapGrid = mergeGrid;

% pcalign crea direttamente la mappa voxelizzata coerente
ptCloudMap = pcalign(ptcs, poses, mapGrid);

end

% --------------------------------------------------------------------------------------------------
function traj = makeTrajectory(poses,R_CL,t_CL)
% Crea interpolatori posizione e orientazione su tempo continuo. pos(t) via interp1 shape pchip;
% quat(t) via slerp su quaternion.

t  = poses.time(:);
N = numel(t);
R_wC = poses.rot;      % 3x3xN
t_wC = poses.pos;      % Nx3
R_wL = zeros(3,3,N);
t_wL = zeros(N,3);
for k = 1:N
    Rwc = R_wC(:,:,k);
    twc = t_wC(k,:).';
    R_wL(:,:,k) = Rwc * R_CL;
    t_wL(k,:)   = (Rwc * t_CL(:) + twc).';
end
traj.R = R_wL;          % 3x3xN   (LiDAR-orientation in world)
traj.p = t_wL;          % Nx3     (LiDAR-position in world)
traj.t = t;
traj.pInterp = @(tt) interp1(t, t_wL, tt, 'linear');
traj.RInterp = @(tt) interp1eul(t, R_wL, tt);

end

% Euler attitude interpolation (rotation vector)
function DD = interp1eul(t, CC, tt)
tt = tt(:);
DD = zeros(3,3,numel(tt));
for i=1:numel(tt)
    if tt(i) <= t(1), DD(:,:,i) = CC(:,:,1); continue; end
    if tt(i) >= t(end), DD(:,:,i) = CC(:,:,end); continue; end
    k = find(t<=tt(i),1,'last');
    t1=t(k); t2=t(k+1);
    % tau = (tt(i)-t1)/(t2-t1);
    C = CC(:,:,k:k+1);
    mu = acos(.5*(C(1,1,:)+C(2,2,:)+C(3,3,:)-1));
    rho = (mu./(2*sin(mu))).*[C(2,3,:)-C(3,2,:); C(3,1,:)-C(1,3,:); C(1,2,:)-C(2,1,:)];
    rho_i = interp1([t1;t2], shiftdim(rho,2), tt, 'linear');
    D = rotmat(quaternion(rho_i,"rotvec"),"frame");
    DD(:,:,i) = D;
end
end

function qOut = localQuatInterp(t, q, tt)
% Interpola quaternioni via SLERP tra due istanti che racchiudono tt.
tt = tt(:);
qOut = quaternion.zeros(numel(tt),1);
for i=1:numel(tt)
    if tt(i) <= t(1), qOut(i) = q(1); continue; end
    if tt(i) >= t(end), qOut(i) = q(end); continue; end
    k = find(t<=tt(i),1,'last');
    t1=t(k); t2=t(k+1);
    tau = (tt(i)-t1)/(t2-t1);
    qOut(i) = slerp(q(k), q(k+1), tau);
end
end

function [R,T] = poseAtTime(traj, tt, k)
% Restituisce R(3x3), T(1x3) alla time tt.
p = traj.pInterp(tt);
% q = traj.qInterp(tt);
% R = rotmat(q,'frame'); % 3x3
% R = traj.RInterp(tt);
R = traj.R(:,:,k);
T = p(:).';            % 1x3
end

function [pc, tpc] = readPointCloudWithTimestamp(filePath)
% Legge point cloud e stima timestamp medio dal nome file o metadata; qui esempio semplificato.
% Formato atteso: ..._t<seconds>.ply  oppure si carica un vettore t esterno.
[~,name,~] = fileparts(filePath);
expr = regexp(name,'_t([0-9]+\.?[0-9]*)','tokens','once');
if ~isempty(expr)
    tpc = str2double(expr{1});
else
    tpc = NaN; % se assente, l'utente deve fornire tempi; può essere gestito con indice k
end
pc = pcread(filePath);
end

function pcOut = transformPointCloud(pcIn, R, T)
% Applica trasformazione SE(3). Se mode=='toWorld' interpreta (R,T) come posa del sensore nel mondo:
% p_world = R*p_local + T.
A = eye(4); A(1:3,1:3)=R; A(1:3,4)=T(:);
tform = affine3d(A');
pcOut = pctransform(pcIn, tform);
end

function pcOut = colorizePointCloud(pcIn, Irect, cameraParams, R_CL, t_CL)
% Attribuisce colore ai punti LiDAR usando immagine rettificata e extrinsics camera<-LiDAR.
% Equazione di proiezione: s [u v 1]^T = K [R_CL|t_CL] [X Y Z 1]^T.
% Usa worldToImage se disponi di cameraParameters; in alternativa projection manuale.

% In questo esempio ipotizziamo che extrinsics siano camera<-LiDAR; portiamo i punti LiDAR nel frame camera.
XYZ = pcIn.Location; % Nx3
XYZc = (R_CL * XYZ.' + t_CL(:)).'; % Nx3
intensity = pcIn.Intensity;

% Proiezione con cameraParams
if isa(cameraParams,'cameraParameters')
    R = eye(3); t = [0 0 0]; % perché XYZc è già nel frame camera
    [uv, ~] = worldToImage(cameraParams, R, t, XYZc);
else
    % cameraIntrinsics (nuove versioni): proiezione manuale
    K = cameraParams.Intrinsics.IntrinsicMatrix.'; % MATLAB usa K' internamente
    XYZn = XYZc ./ XYZc(:,3);
    hom = (K * XYZn.').';
    uv = hom(:,1:2);
end

% Campionamento colore
[h,w,~] = size(Irect);
valid = uv(:,1)>=1 & uv(:,1)<=w & uv(:,2)>=1 & uv(:,2)<=h & XYZc(:,3)>0;
C = zeros(pcIn.Count,3,'uint8');
linIdx = sub2ind([h w], round(uv(valid,2)), round(uv(valid,1)));
Ir = Irect(:,:,1); Ig = Irect(:,:,2); Ib = Irect(:,:,3);
C(valid,1) = Ir(linIdx); 
C(valid,2) = Ig(linIdx); 
C(valid,3) = Ib(linIdx);
C(~valid,1) = intensity(~valid);

pcOut = pointCloud(pcIn.Location, "Color", C, 'Intensity', intensity);
end

function pcOut = processPointCloud(pcIn,gridDown)
    % Segment ground as the dominant plane with reference normal vector pointing in positive
    % z-direction
    maxDistance = 0.2;
    maxAngularDistance = 5;
    referenceVector= [0 0 1];
    % gridDown = 0.05;      % m, downsample per registrazione

    [~,groundFixedIdx] = pcfitplane(pcIn, maxDistance, referenceVector, maxAngularDistance);

    groundFixed = false(pcIn.Count,1);
    groundFixed(groundFixedIdx) = true;

    % Segment ego vehicle as points within a cylindrical region of the sensor
    sensorLocation = [0 0 0];
    egoRadius = 3.5;
    egoFixed = findPointsInCylinder(pcIn,egoRadius,"Center",sensorLocation);
    
    % Retain subset of point cloud without ground and ego vehicle
    indices = find(~groundFixed & ~egoFixed);
    
    pcOut = select(pcIn,indices,'OutputSize','full');
    pcOut = pcdownsample(pcOut, "gridAverage", gridDown);
end

function helperDisplayBoxes(obj,bboxes,labels)
% Display the boxes over the image and point cloud.
    figure
    if ~isa(obj,'pointCloud')
        imshow(obj)
        shape = 'rectangle';
    else
        pcshow(obj.Location);
        shape = 'cuboid';
    end
    showShape(shape,bboxes(labels=='Car',:),...
                  'Color','green','LineWidth',0.5);hold on;
    showShape(shape,bboxes(labels=='Truck',:),...
              'Color','magenta','LineWidth',0.5);
    showShape(shape,bboxes(labels=='Pedestrain',:),...
              'Color','yellow','LineWidth',0.5);
    hold off;
end

function Cout = ortonormal_R(Cin)

% Brute-force orthogonalization, Groves, Eq 5.79
c1 = Cin(:,1);
c2 = Cin(:,2);
c3 = Cin(:,3);
c1 = c1 - 0.5 * (c1'*c2) * c2 - 0.5 * (c1'*c3) * c3 ;
c2 = c2 - 0.5 * (c1'*c2) * c1 - 0.5 * (c2'*c3) * c3 ;
c3 = c3 - 0.5 * (c1'*c3) * c1 - 0.5 * (c2'*c3) * c2 ;

% Brute-force normalization, Groves, Eq 5.80
c1 = c1 / sqrt(c1'*c1);
c2 = c2 / sqrt(c2'*c2);
c3 = c3 / sqrt(c3'*c3);
Cout = [c1 , c2 , c3 ];

end

function pcOut = removePointsInsideBBoxes(pcIn, bboxes3d)
%REMOVEPOINTSINSIDEBBOXES Rimuove i punti di pcIn interni a cuboidi 3D.
%   pcOut = removePointsInsideBBoxes(pcIn, bboxes3d)
%
% Formati supportati per bboxes3d:
% 1) table con variabili {'Centroid','Dimensions','Yaw'} o {'Center','Dimensions','Yaw'}
%    - Centroid/Center: [x y z]
%    - Dimensions: [L W H]
%    - Yaw: scalar (rad) rotazione attorno a Z
% 2) struct array con campi .Center/.Centroid, .Dimensions, .Yaw
% 3) numeric N×10 stile Lidar Toolbox (xmin,ymin,zmin,L,W,H,roll,pitch,yaw,score) → usa yaw, centrando il box
%
% Mantiene Color/Intensity se presenti.

P = pcIn.Location;                    % Nx3
hasColor = ~isempty(pcIn.Color);
if hasColor, C = pcIn.Color; end
hasIntensity = isprop(pcIn,'Intensity') && ~isempty(pcIn.Intensity);
if hasIntensity, I = pcIn.Intensity; end

inside = false(pcIn.Count,1);

centers = []; dims = []; yaws = [];

if istable(bboxes3d)
    cn = bboxes3d.Properties.VariableNames;
    cField = intersect(cn, {'Center','Centroid'});
    assert(~isempty(cField), 'Tabella bbox: manca Center/Centroid.');
    centers = bboxes3d.(cField{1});
    dims    = bboxes3d.Dimensions;
    assert(any(strcmp(cn,'Yaw')), 'Tabella bbox: manca Yaw.');
    yaws    = bboxes3d.Yaw;
elseif isstruct(bboxes3d)
    assert(isfield(bboxes3d,'Dimensions'), 'Struct bbox: manca Dimensions.');
    yaws = extractfield(bboxes3d,'Yaw')';
    if isfield(bboxes3d,'Center')
        centers = vertcat(bboxes3d.Center);
    else
        centers = vertcat(bboxes3d.Centroid);
    end
    dims = vertcat(bboxes3d.Dimensions);
elseif isnumeric(bboxes3d)
    % Assumi formato [xmin,ymin,zmin,L,W,H,roll,pitch,yaw,(score)]
    assert(size(bboxes3d,2)>=9, 'Array bbox numerico atteso N×9 o N×10.');
    L = bboxes3d(:,4); W = bboxes3d(:,5); H = bboxes3d(:,6);
    yaw = bboxes3d(:,9);
    xmin = bboxes3d(:,1); ymin = bboxes3d(:,2); zmin = bboxes3d(:,3);
    centers = [xmin+L/2, ymin+W/2, zmin+H/2];
    dims = [L W H];
    yaws = yaw;
else
    error('Formato bboxes3d non riconosciuto.');
end

Nbox = size(centers,1);
if Nbox==0
    pcOut = pcIn; return;
end

for k = 1:Nbox
    c = centers(k,:);         % 1x3
    d = dims(k,:);            % [L W H]
    psi = yaws(k);            % rad

    % Rotazione Z: Rz(psi); servono coord locali => applica Rz(-psi)
    c2p = P - c;              % N×3, punti centrati
    cp = cos(psi); sp = sin(psi);
    RzT = [ cp  sp  0;       % Rz(-psi) = Rz(psi)^T
           -sp  cp  0;
             0   0  1];
    Ploc = c2p * RzT.';       % N×3 nel frame del box

    half = 0.5*d;
    inK =  abs(Ploc(:,1)) <= half(1) & ...
           abs(Ploc(:,2)) <= half(2) & ...
           abs(Ploc(:,3)) <= half(3);

    inside = inside | inK;
end

keep = ~inside;
if hasColor && hasIntensity
    pcOut = pointCloud(P(keep,:), "Color", C(keep,:), "Intensity", I(keep));
elseif hasColor
    pcOut = pointCloud(P(keep,:), "Color", C(keep,:));
elseif hasIntensity
    pcOut = pointCloud(P(keep,:), "Intensity", I(keep));
else
    pcOut = pointCloud(P(keep,:));
end
end