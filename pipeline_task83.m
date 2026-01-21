% pipeline_task83.m
rng(0);

% 0) Configurazione e caricamento dati
dataDir = "data/task81test/";
pcDir   = dataDir + "lidar/";           % .pcd o .ply o .las (se disponibile)
imgDir  = dataDir + "camera/";          % .png/.jpg
poseFile= dataDir + "imu_gnss/poses.mat"; % struct poses: time [s], position Nx3, orientation quaternion Nx4 [w x y z]
camCal  = dataDir + "cameraParams.mat";  % variabile cameraParams (cameraParameters), salvata da estimateCameraParameters()
extCal  = dataDir + "extrinsics_CL.mat"; % variabile extrCL con campi R_CL (3x3), t_CL (1x3) camera<-LiDAR

pcList  = sort(string(ls(pcDir + "*.p*")));  % supporta .pcd/.ply
imgList = sort(string(ls(imgDir + "*.*g")));  % jpg/png
load(poseFile,"poses");                       % poses.time, poses.pos [N x 3], poses.quat [N x 4] (wxyz)
load(camCal,"cameraParams");                  % cameraParameters o cameraIntrinsics (R2020b+)
load(extCal,"extrCL");                        % extrCL.R_CL, extrCL.t_CL

bevDir = dataDir + "bev/";          % .png/.jpg
bevList = sort(string(ls(bevDir + "*.*g")));  % jpg/png

% Verifica consistenza
assert(~isempty(pcList) && ~isempty(imgList), "Dati LiDAR o immagini mancanti.");
assert(isfield(poses,'time') && isfield(poses,'pos') && isfield(poses,'quat'), "File pose non valido.");
assert(exist('cameraParams','var')==1, "Parametri camera mancanti.");
assert(isfield(extrCL,'R_CL') && isfield(extrCL,'t_CL'), "Extrinsics camera<-LiDAR mancanti.");

% main function --
XX = detect(imgDir+imgList, pcDir+pcList, bevDir+bevList, poses, cameraParams, extrCL.R_CL, extrCL.t_CL);

function XX = detect(imgList, pcList, bevList, poses, cameraParams, R_CL, t_CL)

env = "C:\Users\Michele\anaconda3\envs\DTCyolov9_ultra\python.exe";
pyenv(Version=env);

for i = 1:length(pcList)

    fprintf("Elaborando step %d...\n", i);

    % Read images
    Ik = imread(imgList(i));

    % Read point cloud
    [pcK, tK] = readPointCloudWithTimestamp(pcList(i));
    pcK = colorizePointCloud(pcK, Ik, cameraParams, R_CL, t_CL);

    bb = pyrunfile("yoloe.py","results",img_path=imgList(i));

end

terminate(pyenv)

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
