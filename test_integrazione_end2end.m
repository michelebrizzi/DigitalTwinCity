%% test_integrazione_end2end.m
clear; clc; rng(1);

root = "sandbox_synth/"; if ~isfolder(root), mkdir(root); end
pcDir = root + "data/2025_04_urbano_centro/lidar/";    if ~isfolder(pcDir), mkdir(pcDir); end
imDir = root + "data/2025_04_urbano_centro/camera/";   if ~isfolder(imDir), mkdir(imDir); end
poseDir = root + "data/2025_04_urbano_centro/imu_gnss/"; if ~isfolder(poseDir), mkdir(poseDir); end

% 1) Creo una mappa base con punti su più quote + un "muro"
[x,y] = meshgrid(linspace(-12,12,180));
z = 0.03*randn(numel(x),1);
P0 = [x(:) y(:) z];
wall = [repmat(6,800,1) linspace(-4,4,800).' linspace(0,3,800).'];
P0 = [P0; wall];

% 2) Genero 3 pose con moto rettilineo + piccola rotazione
t = [0; 1.2; 2.5];
T = [0 0 0; 1.0 0.2 0.0; 2.1 0.5 0.05];
ang = [0 0 0; 0 0 deg2rad(2); 0 0 deg2rad(4)]; % rotazioni attorno a Z
R = pagemtimes(eye(3), ones(1,1,3)); % placeholder
for k=1:3
    R(:,:,k) = axang2rotm([0 0 1 ang(k,3)]);
end

% 3) Creo tre scan come trasformazioni inverse della mappa nel frame sensore
pcList = strings(3,1);
for k=1:3
    % world -> sensor_k: p_s = R_k' (p_w - T_k)
    Ps = (R(:,:,k).' * (P0 - T(k,:)).').';
    Ps = Ps + 0.02*randn(size(Ps));
    pc = pointCloud(Ps);
    fname = sprintf("scan_t%.1f.ply", t(k));
    pcwrite(pc, pcDir + fname);
    pcList(k) = pcDir + fname;
end

% 4) Pose file (time, pos, quat[w x y z])
poses.time = t;
poses.pos  = T;
poses.quat = zeros(3,4);
for k=1:3
    q = quaternion(rotm2quat(R(:,:,k))); % returns [w x y z]
    poses.quat(k,:) = compact(q);
end
save(poseDir + "poses.mat","poses");

% 5) Camera intrinseche, extrinseche camera<-LiDAR e immagini "rettificate"
fx=1200; fy=1180; cx=640; cy=360; imgSize=[720 1280];
K = [fx 0 cx; 0 fy cy; 0 0 1];
cameraParams = cameraParameters("IntrinsicMatrix",K',"ImageSize",imgSize);
save(root + "data/2025_04_urbano_centro/cameraParams.mat","cameraParams");

extrCL.R_CL = eye(3);
extrCL.t_CL = [0; 0; 0.2]; % camera 20 cm sopra LiDAR
save(root + "data/2025_04_urbano_centro/extrinsics_CL.mat","extrCL");

% 6) Genero immagini coerenti proiettando la mappa dal frame camera di ciascuna posa
for k=1:3
    % world -> camera_k (assumo camera co-locata al LiDAR salvo offset t_CL in world)
    Cw_R = R(:,:,k); Cw_T = T(k,:) + (Cw_R*extrCL.t_CL).';
    % punti mondo -> camera: Xc = Rcw (Xw - Cw_T)
    Xw = P0; 
    Xc = (Cw_R.' * (Xw - Cw_T)).';
    uv = (K * (Xc(1:3,:)./Xc(3,:)));
    uv = uv(1:2,:).';
    I = uint8(zeros(imgSize(1),imgSize(2),3));
    mask = Xc(3,:)>1 & uv(:,1)>=1 & uv(:,1)<=imgSize(2) & uv(:,2)>=1 & uv(:,2)<=imgSize(1);
    linIdx = sub2ind(imgSize, round(uv(mask,2)), round(uv(mask,1)));
    I(:,:,1) = uint8(zeros(imgSize)); I(:,:,2)=I(:,:,1); I(:,:,3)=I(:,:,1);
    tmp = I(:,:,1); tmp(linIdx)=255; I(:,:,1)=tmp; % punti rossi
    imwrite(I, imDir + sprintf("img_%d_rect.png", k));
end

% 7) Esegue la pipeline puntando alla root sintetica
addpath(pwd);
run pipeline_task82.m
