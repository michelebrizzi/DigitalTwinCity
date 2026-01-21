%% test_unit_moduli.m
clear; clc; rng(0);

% 1) Genero una nuvola sintetica con un piano + punti 3D sparsi
Nplane = 5000; Nobj = 2000;
[x,y] = meshgrid(linspace(-10,10,round(sqrt(Nplane))));
x = x(:); y = y(:);
z = 0.2*randn(numel(x),1); % piano quasi orizzontale z≈0
Pplane = [x y z];
Pobj   = randn(Nobj,3).* [2 2 2] + [0 0 2]; % oggetto sopra il piano
P      = [Pplane; Pobj];
pcA    = pointCloud(P);

% 2) Applico una trasformazione rigida nota per creare pcB
%    R_true = RotZ(10°)*RotY(-5°), t_true = [0.5 -0.3 0.2]
Rz = axang2rotm([0 0 1 deg2rad(10)]);
Ry = axang2rotm([0 1 0 deg2rad(-5)]);
R_true = Rz*Ry;
t_true = [0.5 -0.3 0.2];
P2     = (R_true*P.').'+t_true;
pcB    = pointCloud(P2 + 0.01*randn(size(P2))); % rumore

% 3) ICP: recupero (R_est,t_est) e confronto con verità a terra
[tform, pcB_al, rmse] = pcregistericp(pcB, pcA, "Metric","pointToPlane", ...
    "Tolerance",[1e-4 0.5], "MaxIterations", 80, "Extrapolate",true);

A = tform.T';                   % affine 4x4 (MATLAB memorizza trasposto)
R_est = A(1:3,1:3);
t_est = A(1:3,4).';

% 4) Errori: angolo tra R_est e R_true, norma di t_est - t_true
R_err  = R_est*R_true';
angErr = acos( max(-1,min(1, (trace(R_err)-1)/2 )) ); % [rad]
tErr   = norm(t_est - t_true);

fprintf("ICP: RMSE=%.3f m, angErr=%.3f deg, tErr=%.3f m\n", rmse, rad2deg(angErr), tErr);

% 5) RANSAC per il piano stradale sintetico e verifica residui
maxDistance = 0.05;
[model, inliers, ~] = pcfitplane(pcA, maxDistance, [0 0 1], deg2rad(10));
res = abs(model.Parameters(1)*P(:,1) + model.Parameters(2)*P(:,2) + ...
          model.Parameters(3)*P(:,3) + model.Parameters(4));
fprintf("RANSAC: dist media inlier=%.3f m (atteso ~%.02f m)\n", mean(res(inliers)), maxDistance);

% 6) Camera sintetica + proiezione per test colorazione
fx=1200; fy=1180; cx=640; cy=360; imgSize=[720 1280];
K = [fx 0 cx; 0 fy cy; 0 0 1];
Icolor = uint8(zeros(imgSize(1), imgSize(2), 3));
Icolor(:,:,1) = repmat(uint8(linspace(0,255,imgSize(2))), imgSize(1), 1);
Icolor(:,:,2) = repmat(uint8(linspace(255,0,imgSize(2))), imgSize(1), 1);
Icolor(:,:,3) = 128;

% Extrinsics camera<-LiDAR sintetiche
R_CL = eye(3); t_CL = [0 0 0].';

XYZc = (R_CL*pcA.Location.' + t_CL); XYZc = XYZc.';
uvh  = (K * (XYZc./XYZc(:,3)).').'; % s*[u v 1]^T = K*[X/Z Y/Z 1]^T
uv   = uvh(:,1:2);
h=imgSize(1); w=imgSize(2);
valid = uv(:,1)>=1 & uv(:,1)<=w & uv(:,2)>=1 & uv(:,2)<=h & XYZc(:,3)>0;
C = zeros(pcA.Count,3,'uint8');
linIdx = sub2ind([h w], round(uv(valid,2)), round(uv(valid,1)));
Ir = Icolor(:,:,1); Ig = Icolor(:,:,2); Ib = Icolor(:,:,3);
C(valid,1)=Ir(linIdx); C(valid,2)=Ig(linIdx); C(valid,3)=Ib(linIdx);
pcColor = pointCloud(pcA.Location, "Color", C);

pcshow(pcColor); title("Colorazione sintetica: gradiente rosso-verde");
