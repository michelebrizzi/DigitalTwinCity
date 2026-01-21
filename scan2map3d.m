function ptCloudMap = scan2map3d(imgList, pcList, bevList, poses, cameraParams, R_CL, t_CL)

% Parameters for Point Cloud Registration Algorithm Specify the parameters for estimating the
% trajectory using the point cloud registration algorithm. 
maxLidarRange = 50;             % maximum range of the 3-D laser scanner
referenceVector = [0 0 1];      % Normal to the ground plane.
maxDistance = 0.5;              % Maximum distance for inliers when removing the ground plane
maxAngularDistance = 15;        % Maximum angle deviation from the reference vector when fitting the ground and ceiling planes.
randomSampleRatio = 0.1;       % The point clouds are downsampled using random sampling with a sample ratio specified by randomSampleRatio.
gridStep = 2;                 % gridStep specifies the voxel grid sizes used in the NDT registration algorithm. 
distanceMovedThreshold = 0.3;   % A scan is accepted only after the robot moves by a distance greater than distanceMovedThreshold.

% Parameters for Loop Closure Estimation Algorithm. 
loopClosureSearchRadius = 3;    % Loop closures are only searched within a radius around the current robot location
nScansPerSubmap = 3;            % A submap is created after every nScansPerSubmap (Number of Scans per submap) accepted scans. 
subMapThresh = 50;              % The loop closure algorithm disregards the most recent subMapThresh scans while searching for loop candidates.

% The maximum acceptable Root Mean Squared Error (RMSE) in estimating relative pose between loop candidates is specified by rmseThreshold. Choose a lower value for estimating accurate loop closure edges, which has a high impact on pose graph optimization.
rmseThreshold = 0.25;
loopClosureThreshold = 150;     % The threshold over scan matching score to accept a loop closure
optimizationInterval = 2;       % Pose Graph Optimization is called after inserting optimizationInterval loop closure edges into the pose graph.

% Carica detector pre-addestrato
useDetector = 0;
if useDetector
    % detector = voxelRCNNObjectDetector("kitti");
    % detector = load("models/complex-yolov4/cspdarknet-complex-yolov4.mat").detector;
    detector = load('pretrainedPointPillarsDetector.mat','detector').detector;
end

% Inizializzazioni empty viewset
vSet = pcviewset;

% Set up a pose graph, occupancy map, and necessary variables.
pGraph = poseGraph3D;   % 3D Posegraph object for storing estimated relative poses
infoMat = [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1];% Default serialized upper-right triangle of 6-by-6 Information Matrix
numLoopClosuresSinceLastOptimization = 0;% Number of loop closure edges added since last pose graph optimization and map refinement
mapUpdated = false; % True after pose graph optimization until the next scan
scanAccepted = 0; % Equals to 1 if the scan is accepted
gridDown = .05; % m, downsample per registrazione
traj = makeTrajectory(poses,R_CL,t_CL); % struct con interpPos(t), interpQuat(t)

% 3D Occupancy grid object for creating and visualizing 3D map
mapResolution = 8; % cells per meter
omap = occupancyMap3D(mapResolution);

% Preallocate variables for the processed point clouds, lidar scans, and submaps. Create a downsampled set of point clouds for quickly visualizing the map.
pcProcessed = cell(1,length(pcList));
lidarScans2d = cell(1,length(pcList));
submaps = cell(1,length(pcList)/nScansPerSubmap);
pcsToView = cell(1,length(pcList));

% View the point clouds while processing them sequentially
viewPC = 0; % Set to 1 to visualize processed point clouds during build process
if viewPC==1
    pplayer = pcplayer([-50 50],[-50 50],[-10 10],'MarkerSize',5);
end

% View the created map and posegraph during build process
viewMap = 0; % Set to 1 to visualize created map and posegraph during build process
if viewMap==1
    ax = newplot;
    view(20,50);
    grid on;
end

% Initialize transformations
absTform = rigidtform3d;  % Absolute transformation to reference frame
prevTform = rigidtform3d;  % Relative transformation between successive scans
viewId = 1;

% Trajectory Estimation and Refinement Using Pose Graph Optimization The trajectory of the robot is
% a collection of robot poses (location and orientation in 3-D space). A robot pose is estimated at
% every 3-D lidar scan acquisition instance using the 3-D lidar SLAM algorithm. Iteratively process
% the point clouds to estimate the trajectory.
count = 0; % Counter to track number of scans added
% disp('Estimating robot trajectory...');
for i = 1:length(pcList)

    fprintf("Elaborando step %d...\n", i);

    % Read images
    Ik = imread(imgList(i));

    % Read point cloud
    [pcK, tK] = readPointCloudWithTimestamp(pcList(i));
    pcK = colorizePointCloud(pcK, Ik, cameraParams, R_CL, t_CL);
    
    % Process point cloud
    %   - Segment and remove ground plane
    %   - Segment and remove ego vehicle
    %   - Downsample point cloud
    pcl_wogrd_sampled = processPointCloud(pcK,gridDown);

    % Read BEV
    % bevk = imread(bevList(i));

    % Read poses
    % [Rk_gt, Tk_gt] = poseAtTime(traj, tK, i);

    if viewPC==1
        % Visualize down sampled point cloud
        view(pplayer,pcl_wogrd_sampled);
        pause(0.001)
    end

    % Point Cloud Registration ---------------------------------------------------------------------
    % Point cloud registration estimates the relative pose (rotation and translation) between
    % current scan and previous scan. The first scan is always accepted (processed further and
    % stored) but the other scans are only accepted after translating more than the specified
    % threshold. poseGraph3D is used to store the estimated accepted relative poses (trajectory).
    if count == 0
        % First scan
        tform = [];
        scanAccepted = 1;
        
        % Add first point cloud scan as a view to the view set
        vSet = addView(vSet,viewId,absTform,"PointCloud",pcK);
    else
        tform = pcregisterndt(pcl_wogrd_sampled,prevPc,gridStep,'InitialTransform',prevTform);

        relPose = [tform2trvec(tform.T') tform2quat(tform.T')];
        if sqrt(norm(relPose(1:3))) > distanceMovedThreshold
            addRelativePose(pGraph,relPose);
            scanAccepted = 1;

            % Update absolute transformation to reference frame (first point cloud)
            absTform = rigidtform3d(absTform.A * tform.A);

            % Add current point cloud scan as a view to the view set
            vSet = addView(vSet,viewId,absTform,"PointCloud",pcK);
            vSet = addConnection(vSet,viewId-1,viewId,tform);
        else
            scanAccepted = 0;
        end
    end

    % Loop Closure Query ---------------------------------------------------------------------------
    % Loop closure query determines whether or not the current robot location has previously been
    % visited. The search is performed by matching the current scan with the previous scans within a
    % small radius around the current robot location specified by loopClosureSearchRadius. Searching
    % within the small radius is sufficient because of the low-drift in lidar odometry, as searching
    % against all previous scans is time consuming. Loop closure query consists of the following
    % steps: Create submaps from nScansPerSubmap consecutive scans. Match the current scan with the
    % submaps within the loopClosureSearchRadius. Accept the matches if the match score is greater
    % than the loopClosureThreshold. All the scans representing accepted submap are considered as
    % probable loop candidates. Estimate the relative pose between probable loop candidates and the
    % current scan. A relative pose is accepted as a loop closure constraint only when the RMSE is
    % less than the rmseThreshold.
    if scanAccepted == 1
        count = count + 1;

        % detection 3D su non-ground e masking dinamici
        if useDetector
            try
                % [bboxes3d, scores, labels] = detect(detector, pcNG, "MinScore", 0.4); %voxelnetrcnn
                % [bboxes, scores, labels] = detect(detector, bevk); % complex-yolov4
                % [ptCldOut,bboxCuboid] = transferbboxToPointCloud(bboxes,gridParams,pcK);
                [bboxes,score,labels] = detect(detector,pcl_wogrd_sampled); idx = score>.65;
                % helperDisplayBoxes(pcK,bboxes(idx,:),labels(idx));
                pcl_wogrd_sampled = removePointsInsideBBoxes(pcl_wogrd_sampled, bboxes(idx,:));
            catch
                % prosegui senza detection
            end
        end

        pcProcessed{count} = pcl_wogrd_sampled;
        lidarScans2d{count} = create2DScan(pcl_wogrd_sampled);

        % Submaps are created for faster loop closure query.
        if rem(count,nScansPerSubmap)==0
            submaps{count/nScansPerSubmap} = exampleHelperCreateSubmap(lidarScans2d,pGraph,count,nScansPerSubmap,maxLidarRange);
        end

        % loopSubmapIds contains matching submap ids if any otherwise empty.
        if (floor(count/nScansPerSubmap) > subMapThresh)
            [loopSubmapIds,~] = estimateLoopCandidates(pGraph, count,submaps,lidarScans2d{count},nScansPerSubmap, loopClosureSearchRadius,loopClosureThreshold,subMapThresh);

            if ~isempty(loopSubmapIds)
                rmseMin = inf;

                % Estimate best match to the current scan
                for k = 1:length(loopSubmapIds)
                    % For every scan within the submap
                    for j = 1:nScansPerSubmap
                        probableLoopCandidate = loopSubmapIds(k)*nScansPerSubmap - j + 1;
                        [loopTform,~,rmse] = pcregisterndt(pcl_wogrd_sampled, pcProcessed{probableLoopCandidate},gridStep);
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

                    addRelativePose(pGraph, relPose, infoMat, loopCandidate, count);
                    numLoopClosuresSinceLastOptimization = numLoopClosuresSinceLastOptimization + 1;

                    % Add a connection corresponding to a loop closure
                    vSet = addConnection(vSet,loopViewId,viewId,loopTform,infoMat);
                end

            end
        end

        % Pose graph optimization runs after a sufficient number of loop edges are accepted to
        % reduce the drift in trajectory estimation. After every loop closure optimization the loop
        % closure search radius is reduced due to the fact that the uncertainty in the pose
        % estimation reduces after optimization.
        if (numLoopClosuresSinceLastOptimization == optimizationInterval)|| ((numLoopClosuresSinceLastOptimization>0)&&(i==length(pcList)))
            if loopClosureSearchRadius ~=1
                disp('Doing Pose Graph Optimization to reduce drift.');
            end
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
        % Visualize the map and pose graph during the build process. This visualization is costly, so enable it only when necessary by setting viewMap to 1. If visualization is enabled then the plot is updated after every 15 added scans.
        pcToView = pcdownsample(pcl_wogrd_sampled, 'random', 0.5);
        pcsToView{count} = pcToView;

        if viewMap==1
            % Insert point cloud to the occupancy map in the right position
            position = pGraph.nodes(count);
            insertPointCloud(omap,position,pcToView.removeInvalidPoints,maxLidarRange);

            if (rem(count-1,15)==0)||mapUpdated
                visualizeMapAndPoseGraph(omap, pGraph, ax);
            end
            mapUpdated = false;
        else
            % Give feedback to know that example is running
            if (rem(count-1,15)==0)
                fprintf('.');
            end
        end

        % Update previous relative pose estimate and point cloud.
        prevPc = pcl_wogrd_sampled;
        prevTform = tform;
        viewId = viewId + 1;
    end
end

% Build and Visualize 3-D Occupancy Map ------------------------------------------------------------
% The point clouds are inserted into occupancyMap3D using the estimated global poses. After
% iterating through all the nodes, the full map and estimated vehicle trajectory is shown.
if (viewMap ~= 1)||(numLoopClosuresSinceLastOptimization>0)
    nodesPositions = nodes(pGraph);
    
    optimG = optimizePoseGraph(pGraph,'g2o-levenberg-marquardt');
    vSetOptim = updateView(vSet,optimG.Nodes);
    mapGridSize = 0.2;
    ptClouds = vSetOptim.Views.PointCloud;
    absPoses = vSetOptim.Views.AbsolutePose;
    ptCloudMap = pcalign(ptClouds,absPoses,mapGridSize);

    figure
    plot(vSetOptim)
    hold on
    plot(vSetOptim);
    title('Point Cloud Map (after optimization)','Color','w')

    % Create 3D Occupancy grid
    omapToView = occupancyMap3D(mapResolution);

    for i = 1:(size(nodesPositions,1)-1)
        pc = pcsToView{i};
        position = nodesPositions(i,:);

        % Insert point cloud to the occupancy map in the right position
        insertPointCloud(omapToView,position,pc.removeInvalidPoints,maxLidarRange);
    end

    figure;
    axisFinal = newplot;
    visualizeMapAndPoseGraph(omapToView, pGraph, axisFinal);
end

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

function lidarScan2d = create2DScan(ptCloud)
% This helper function is used to create 2D Lidar scans from 3D Lidar Scans. These 2D scans are
% required for 2D submap creation which are useful for loop closure query. 3D Point Cloud Down
% sampling is done before creating a 2D scan to reduce the computations. annularSamplingRatio
% specifies the sample ratio to uniform sample the annular region. The sampling ratio is empirically
% chosen for this example.

annularSamplingRatio = 0.1; % Ratio used to sample extracted  3D point cloud annular region

% Use only a subset of the points randomly sampled, since we will have too
% many points if not
ptCloudDownSampled = pcdownsample(ptCloud, 'random', annularSamplingRatio);
loc = ptCloudDownSampled.Location;

% Get the cartesian co-ordinates of the accepted points
cart = double(loc(:,1:2));

lidarScan2d = lidarScan(cart);
end

function [submap] = exampleHelperCreateSubmap(lidarScans,poseGraph,currentId,nScansPerSubmap,maxRange)
% This helper function is used to create 2D submaps from 2D Lidar Scans which have been created by
% slicing the point clouds along an annular region. These Submaps are required for detecting loop
% closures. Each submap represents multiple scans.

submapResolution = 3; % Submap resolution
submapMaxLevel = 3; % Maximum Grid levels 

% Initialize variables
scanIndices = (currentId-nScansPerSubmap+1) : currentId; % Scans to be added to the submap
poses3D = poseGraph.nodes(scanIndices); % Poses of these scans
anchorIndex = round(nScansPerSubmap/2); % Anchor index for the Submap

% Find the 2D position for this submap
poses2D = zeros(nScansPerSubmap,3);
lScans = cell(1,nScansPerSubmap);
for i = 1:nScansPerSubmap
    poses2D(i,1:2) = poses3D(i,1:2);
    eu = quat2eul(poses3D(i,4:end));
    poses2D(i,3) = eu(1);
    lScans{i} = lidarScans{scanIndices(i)};
end

% Create the submap
submap = nav.algs.internal.createSubmap(lScans, 1:nScansPerSubmap, poses2D, anchorIndex, submapResolution, maxRange, submapMaxLevel);
end

function visualizeMapAndPoseGraph(omap, pGraph, ax)
% This helper function is useful for visualizing the built occupancy map 3D (omap) and pose graph
% (pGraph). The plot view is tuned for this example.

show(omap,'Parent',ax);
hold on;
pGraph.show('Parent',ax,"IDs","off");
xlim([-50 50]);
ylim([-50 50]);
zlim([-20 20]);
view([20,50]);
drawnow
hold off;
grid on;
end

function [loopSubmapIds,loopScores] = estimateLoopCandidates(pGraph,currentScanId,submaps,currScan, nScansPerSubmap,loopClosureSearchRadius,loopClosureThreshold,subMapThresh)
% This helper function to returns submap ids which lie within a radius from current scan and match
% with the current scan.Instead of matching the current scan with all the previously accepted scans
% for faster query current scan is matched against a submap (group of scans). Due to this the number
% of matching operation reduces significantly. The submaps are said to be matching with the current
% scan when the submap and scan match score is greater than loopClosureThreshold. Most recent
% subMapThresh submaps are not considered while estimating a loop sub map.

loopClosureMaxAttempts = 8; % Number of submaps checked
maxLoopClosureCandidates = 2; % Number of matches to send back, if matches are found

% Initialize variables
loopSubmapIds = []; % Variable to hold the candidate submap IDs 
loopScores = []; % Variable to hold the score of the matched submaps
pose3D = pGraph.nodes(currentScanId); % Pose of the current node
currentSubMapId = floor((currentScanId-1)/nScansPerSubmap)+1; % Submap corresponding to the node

% Find the submaps to be considered based on distance and Submap ID
% Find the most recent scan center
mostRecentScanCenter = zeros(1,3); 
mostRecentScanCenter(1:2) = pose3D(1:2);
eulAngles = quat2eul(pose3D(4:end));
mostRecentScanCenter(3) = eulAngles(1);

% Compute the centers of all Submaps 
nsbmp = floor(pGraph.NumNodes/nScansPerSubmap);
centers = zeros(nsbmp-1, 2);
for i = 1:nsbmp-1 % ignore the most recent submap
    centers(i,:) = submaps{i}.Center;
end

% Compute the distance of all submaps from the current scan
centerCandidates = zeros(nsbmp-1,2);
n=0; % To keep track of the number of candidates added
for i = 1:nsbmp-1
    distanceToCenters = norm(centers(i, :) - mostRecentScanCenter(1:2));
    % Accept the submap only if it is within the search radius and if its 
    % ID is above the submap threshold w.r.t the current submap
    if (distanceToCenters < loopClosureSearchRadius)&&(abs(i-currentSubMapId)>subMapThresh)
        n = n+1; % Increase the number of candidates added
        centerCandidates(n,:) = [distanceToCenters i]; % Distance and the Submap ID
    end
end
% Only keep the candidates added
centerCandidates = centerCandidates(1:n,:);

% If there are submaps to be considered, sort them by distance from the
% current scan
if ~isempty(centerCandidates)
    % Sort them based on the distance from the current scan
    centerCandidates = sortrows(centerCandidates);
    
    % Return only the minimum number of loop candidates to be returned
    N = min(loopClosureMaxAttempts, size(centerCandidates,1));
    nearbySubmapIDs = centerCandidates(1:N, 2)';
else
    nearbySubmapIDs = [];
end

% Match the current scan with the candidate submaps
newLoopCandidates = zeros(1000,1); % Loop candidates
newLoopCandidateScores = zeros(1000,1); % Loop candidate scores
count = 0; % Number of loop closure matches found

% If there are submaps to consider
if ~isempty(nearbySubmapIDs)
    % For every candidate submap 
    for k = 1:length(nearbySubmapIDs)
        submapId = nearbySubmapIDs(k);
        
        % Match the scan with the submap
        [~, score, ~] = nav.algs.internal.matchScansGridSubmap(currScan, submaps{submapId}, 0, [0 0 0], [0 0], 0); 
        
        % Accept submap only if it meets the score threshold
        if score > loopClosureThreshold 
            count = count + 1;
            % Keep track of matched Submaps and their scores
            newLoopCandidates(count) = submapId;
            newLoopCandidateScores(count) = score;
        end
    end
    
    % If there are candidates to consider
    if ~isempty(newLoopCandidates)
        % Sort them by their scores in descending order
        [~,ord] = sort(newLoopCandidateScores,'descend');
        % Return the required number of submaps matched, and their scores
        loopSubmapIds = newLoopCandidates(ord(1:min(count,maxLoopClosureCandidates)));
        loopScores = newLoopCandidateScores(ord(1:min(count,maxLoopClosureCandidates)));  
    end
end
end



