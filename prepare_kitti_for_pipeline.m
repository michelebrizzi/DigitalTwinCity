%% prepare_kitti_for_pipeline.m
% Adatta KITTI "odometry" o "raw" alla pipeline MATLAB fornita.
% Imposta le seguenti path alla tua installazione locale KITTI.

clear; clc;

% Scegli dataset: 'odometry' (sequence 00..10 con ground truth pose) oppure 'raw'
datasetKind = "odometry";      % "odometry" oppure "raw"
seqId       = "00";            % es. "00" per odometry train; per raw usa data string se serve
maxFrames   = 3000;             % limita per test veloce; [] = tutti

% PATH di origine KITTI (modifica qui)
rootKitti = "D:\OneDrive - Universita degli Studi Roma Tre\Datasets\KITTI";
if datasetKind=="odometry"
    kittiSeqDir = fullfile(rootKitti, "odometry", "dataset");
    velDir  = fullfile(kittiSeqDir, "sequences", seqId, "velodyne");
    imgDir2 = fullfile(kittiSeqDir, "sequences", seqId, "image_2"); % left color rect
    calibFile = fullfile(kittiSeqDir, "sequences", seqId, "calib.txt");
    posesFile = fullfile(kittiSeqDir, "poses", seqId + ".txt"); % ground truth per training set
    assert(isfile(posesFile), "File poses mancante: usa seq 00..10 (training) oppure passa a datasetKind='raw'.");
elseif datasetKind=="raw"
    % Esempio: raw date, drive e sync già eseguito
    dateStr   = "2011_09_26";
    driveStr  = "2011_09_26_drive_0001_sync";
    baseRaw   = fullfile(rootKitti, "raw", dateStr, driveStr);
    velDir    = fullfile(baseRaw, "velodyne_points", "data");
    imgDir2   = fullfile(baseRaw, "image_02", "data");
    calibFile = fullfile(rootKitti, "raw", dateStr, "calib_cam_to_cam.txt");
    velo2cam  = fullfile(rootKitti, "raw", dateStr, "calib_velo_to_cam.txt");
    oxtsDir   = fullfile(baseRaw, "oxts", "data"); % per stimare pose
else
    error("datasetKind sconosciuto.");
end

% PATH di destinazione per la pipeline
destRoot = fullfile(pwd, "data", "task81test");
destLiDAR = fullfile(destRoot, "lidar");     if ~isfolder(destLiDAR), mkdir(destLiDAR); end
destCam   = fullfile(destRoot, "camera");    if ~isfolder(destCam), mkdir(destCam); end
destPose  = fullfile(destRoot, "imu_gnss");  if ~isfolder(destPose), mkdir(destPose); end
destBEV  = fullfile(destRoot, "bev");  if ~isfolder(destBEV), mkdir(destBEV); end

binList = dir(fullfile(velDir, "*.bin"));
if isempty(binList), error("Nessun .bin in %s", velDir); end
nFrames = numel(binList);
if ~isempty(maxFrames), nFrames = min(nFrames, maxFrames); end

% 1) Legge calibrazioni KITTI e ricava intrinseche ed extrinseche camera<-LiDAR
% Odometry: calib.txt contiene P0..P3, R0_rect, Tr_velo_to_cam
% Raw: usare i file calib_cam_to_cam.txt e calib_velo_to_cam.txt
if datasetKind=="odometry"
    calib = readKittiOdometryCalib(calibFile);
    % intrinseche dalla camera 2: P2 = K2 [R|t] con R=I rettificata; K da P2(1:3,1:3)
    K = calib.P2(1:3,1:3);
    imageSize = inferImageSize(imgDir2);
    cameraParams = cameraParameters("IntrinsicMatrix",K',"ImageSize",imageSize);
    % extrinseche camera<-LiDAR: T = R_rect * Tr_velo_to_cam
    T_cam_velo = calib.R0_rect * calib.Tr_velo_to_cam;
    R_CL = T_cam_velo(1:3,1:3);
    t_CL = T_cam_velo(1:3,4);
elseif datasetKind=="raw"
    [K, R_rect, T_cam_velo, imageSize] = readKittiRawCalib(calibFile, velo2cam);
    cameraParams = cameraParameters("IntrinsicMatrix",K',"ImageSize",imageSize);
    R_CL = (R_rect * T_cam_velo(1:3,1:3));
    t_CL = (R_rect * T_cam_velo(1:3,4));
end
save(fullfile(destRoot, "cameraParams.mat"), "cameraParams");
extrCL.R_CL = R_CL; extrCL.t_CL = t_CL;
save(fullfile(destRoot, "extrinsics_CL.mat"), "extrCL");

% 2) Pose: usa ground truth odometry (train) oppure stima da OXTS per raw
if datasetKind=="odometry"
    T_w_c = readKittiOdometryPoses(posesFile);  % Mx4x4
    % Allineiamo i frame presenti; se nFrames < M, tronchiamo
    M = min(size(T_w_c,3), nFrames);
    poses.time = (0:M-1).'/10;
    poses.pos  = squeeze(T_w_c(1:3,4,1:M)).';
    poses.pos  = poses.pos(:,1:3);
    poses.rot = T_w_c(1:3,1:3,1:M);
    poses.quat = zeros(M,4);
    for k=1:M
        Rw = T_w_c(1:3,1:3,k);
        q = quaternion(rotm2quat(Rw)); poses.quat(k,:) = compact(q);
    end
elseif datasetKind=="raw"
    % Stima rapida da OXTS integrando velocità nel frame di mondo.
    % Equazioni: p_{k+1} = p_k + R(\psi_k,\phi_k,\theta_k) v_k \Delta t, con v_k longitudinale
    % Per test: integrazione semplice, dt dal timestamp file.
    [poses.time, poses.pos, poses.quat] = oxtsToPoses(oxtsDir);
end
save(fullfile(destPose, "poses.mat"), "poses");

% 3) Converte scansioni Velodyne .bin in .ply con timestamp a 10 Hz: t = k / 10
fprintf("Converto %d scansioni Velodyne...\n", nFrames);
for k=1:nFrames
    fileBin = fullfile(velDir, binList(k).name);
    [pts,r] = readKittiBinXYZ(fileBin);          % Nx3
    pc = pointCloud(pts,'Intensity',r);
    t = (k-1)/10;                            % KITTI odometry tipicamente 10 Hz
    outName = sprintf("seq%s_t%07.3f.ply", seqId, t);
    pcwrite(pc, fullfile(destLiDAR, outName));
end

% 4) Genera immagini Birds-Eye View (BEV)
pcds = fileDatastore(destLiDAR,'ReadFcn',@(x) pcread(x));
xMin = -50.0;     
xMax = 50.0;      
yMin = -50.0;      
yMax = 50.0;      
zMin = -7.0;     
zMax = 15.0;  
bevHeight = 608;
bevWidth = 608;
gridW = (yMax - yMin)/bevWidth;
gridH = (xMax - xMin)/bevHeight;
gridParams = {{xMin,xMax,yMin,yMax,zMin,zMax},{bevWidth,bevHeight},{gridW,gridH}};
transformPCtoBEV(pcds,gridParams,destBEV);

% 5) Copia o genera immagini rettificate. KITTI odometry image_2 è già rectificata.
pngList = dir(fullfile(imgDir2, "*.png")); if isempty(pngList), pngList = dir(fullfile(imgDir2, "*.jpg")); end
assert(~isempty(pngList), "Nessuna immagine in %s", imgDir2);
nImg = min(numel(pngList), nFrames);
fprintf("Copio %d immagini...\n", nImg);
for k=1:nImg
    src = fullfile(imgDir2, pngList(k).name);
    dst = fullfile(destCam, sprintf("img_%06d_rect.png", k));
    I = imread(src);
    if datasetKind=="odometry"
        Iu = I;
    elseif datasetKind=="raw"
        Iu = undistortImage(I, cameraParams, "OutputView","full"); % già rect ma coerente con pipeline
    end
    imwrite(Iu, dst);
end

fprintf("Preparazione completata in %s\n", destRoot);

function imageSize = inferImageSize(imgDir)
    L = dir(fullfile(imgDir, "*.png")); if isempty(L), L = dir(fullfile(imgDir, "*.jpg")); end
    I = imread(fullfile(imgDir, L(1).name));
    imageSize = [size(I,1) size(I,2)];
end

%% kitti_utils.m
function calib = readKittiOdometryCalib(calibFile)
% Legge calib.txt dell'odometry set: P0..P3, R0_rect, Tr_velo_to_cam
    T = readlines(calibFile);
    calib = struct();
    for i=1:numel(T)
        if strlength(T(i))==0, continue; end
        parts = split(T(i), ':');
        key = strtrim(parts(1)); vals = str2double(split(strtrim(parts(2))));
        switch key
            case {'P0','P1','P2','P3'}
                calib.(key) = reshape(vals, [4,3])'; % 3x4
            case 'R0_rect'
                calib.R0_rect = reshape(vals, [3,3])';
            case 'Tr'
                M = reshape(vals, [4,3])'; % 3x4
                calib.Tr_velo_to_cam = M;
        end
    end
    %assert(isfield(calib,'P2') && isfield(calib,'R0_rect') && isfield(calib,'Tr_velo_to_cam'),"Calibrazione incompleta.");
    if ~isfield(calib,'R0_rect')
        calib.R0_rect = eye(3);
    end
end

function [K, R_rect, T_cam_velo, imageSize] = readKittiRawCalib(cam2camFile, velo2camFile)
% Legge calibrazioni RAW. Usa P_rect_02 come K* [I|0] e R_rect_02 per rettifica.
    C = readCalibKV(cam2camFile);
    V = readCalibKV(velo2camFile);
    P2 = reshape(str2double(split(C.P_rect_02)), [4,3])';
    K = P2(1:3,1:3);
    R_rect = reshape(str2double(split(C.R_rect_02)), [3,3])';
    Tr = reshape(str2double(split(V.Tr_velo_to_cam)), [4,3])';
    T_cam_velo = Tr; % 3x4
    % imageSize = [str2double(C.S_rect_02(5:end)) str2double(split(C.S_rect_02(1:3),'x'){2})];
    % Nota: per RAW l'estrazione imageSize può variare; al bisogno, leggi un file immagine.
    imageSize = []; % fallback → inferito da immagine
end

function S = readCalibKV(fn)
    R = readlines(fn); S = struct();
    for i=1:numel(R)
        if strlength(R(i))==0, continue; end
        parts = split(R(i),':');
        key = strtrim(parts(1)); val = strtrim(parts(2));
        S.(key) = val;
    end
end

function [pts,r] = readKittiBinXYZ(fileBin)
% Legge un .bin KITTI Velodyne [x y z r] float32 e restituisce Nx3.
    fid = fopen(fileBin,'rb');
    A = fread(fid,[4, inf],'single')'; fclose(fid);
    pts = A(:,1:3);
    r = A(:,4);
end

function T = readKittiOdometryPoses(posesFile)
% Ritorna T(:,:,k) con trasformazioni 4x4 camera-pose in mondo per ogni frame.
    A = readmatrix(posesFile);
    n = size(A,1);
    T = zeros(4,4,n);
    for k=1:n
        M = reshape(A(k,:), [4,3])';
        T(:,:,k) = [M; 0 0 0 1];
    end
end

function [time, pos, quatWXYZ] = oxtsToPoses(oxtsDir)
% Stima rapida pose da OXTS RAW. Integra velocità longitudinale con yaw per test.
% Modello: p_{k+1} = p_k + R_z(psi_k) [v_k 0 0]^T dt. Orientazione da yaw, pitch, roll.
    files = dir(fullfile(oxtsDir, "*.txt"));
    n = numel(files); assert(n>0, "Nessun OXTS trovato.");
    time = zeros(n,1); pos = zeros(n,3); quatWXYZ = zeros(n,4);
    yaw=0; pitch=0; roll=0; p=[0 0 0]; tprev=0;
    for k=1:n
        A = readmatrix(fullfile(oxtsDir, files(k).name));
        % Formato OXTS: ... velo forward (vf) = A(9); yaw= A(6), pitch= A(5), roll= A(4) in alcune versioni
        % Attenzione: questo parser è un placeholder, adatta ai tuoi file OXTS specifici.
        vf = A(9); yaw = A(6); pitch = A(5); roll = A(4);
        if k==1, dt=0; else, dt = 0.1; end % 10 Hz approssimato
        Rz = axang2rotm([0 0 1 yaw]);
        v_world = (Rz * [vf;0;0])*dt;
        p = p + v_world.';
        time(k) = (k-1)*0.1;
        pos(k,:) = p;
        q = quaternion(eul2quat([yaw pitch roll])); % ZYX se usi eul2quat default (adatta se necessario)
        quatWXYZ(k,:) = compact(q);
    end
end
