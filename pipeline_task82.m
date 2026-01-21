% pipeline_task82.m
% Pipeline completa per: sincronizzazione, rettifica immagini, compensazione del moto LiDAR,
% registrazione camera-LiDAR, ICP/SLAM, rimozione piano stradale, clustering, colorazione point cloud.
clear; clc;
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
mapPC = scan2map3d(imgDir+imgList, pcDir+pcList, bevDir+bevList, poses, cameraParams, extrCL.R_CL, extrCL.t_CL);
pcshow(mapPC)

voxelSize = 1;
ndtMap = pcmapndt(mapPC,voxelSize);
figure, show(ndtMap)
view(2) 

% 7) Salvataggio risultati
outDir = dataDir + "fused/";
if ~isfolder(outDir), mkdir(outDir); end
pcwrite(mapPC, outDir + "map_fused_colorized.ply");        % mappa completa
fprintf("Completato.\n");