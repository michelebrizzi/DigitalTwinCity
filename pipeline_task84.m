classes_num = [0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 52,...
    60, 70, 71, 72, 80, 81, 99, 252, 253, 254, 255, 256, 257, 258. 259];
class_name = {"unlabeled","outlier","car","bicycle","bus","motorcycle","on-rails","truck","other-vehicle",...
"person","bicyclist","motorcyclist","road","parking","sidewalk","other-ground","building","fence",...
"other-structure","lane-marking","vegetation","trunk","terrain","pole","traffic-sign","other-object",...
"moving-car","moving-bicyclist","moving-person","moving-motorcyclist","moving-on-rails","moving-bus",...
"moving-truck","moving-other-vehicle"};


% "thing" ids in SemanticKITTI (sem_id)
thingIds = [ ...
    10 11 13 15 16 18 20 ...  % vehicles / bikes
    30 31 32 % persone e rider
];

% Path al dataset SemanticKITTI
baseDir = "D:\OneDrive - Universita degli Studi Roma Tre\Datasets\KITTI\odometry\dataset";
seq     = '00';                                   % sequenza da usare
calibFile = sprintf('%s/sequences/%s/calib.txt', baseDir, seq);
poseFile  = sprintf('%s/poses/%s.txt', baseDir, seq);
T_w_cam0 = readKittiOdometryPoses(poseFile);
T_cam0_velo = readKittiCalibTr(calibFile);
useWorld = true; % or false if you want LiDAR-local

frameIds = 0:500;  % primi 200 frame, ad esempio
instanceInfo = analyzeInstancePresence(baseDir, seq, frameIds, classes_num, class_name, thingIds);

% Instances that disappear and reappear:
idxGappy = find([instanceInfo.hasGaps]);
gappyInstances = instanceInfo(idxGappy);

for k = 1:numel(gappyInstances)
    instId    = gappyInstances(k).id;
    frameList = gappyInstances(k).frames;

    scanId1 = frameList(1);   % primo istante
    scanId2 = frameList(end);  % secondo istante
    f1 = scanId1 + 1;   % gli indici in poses partono da 1
    f2 = scanId2 + 1;
    T_w_velo_1 = T_w_cam0{f1} * T_cam0_velo;
    T_w_velo_2 = T_w_cam0{f2} * T_cam0_velo;

    binFile1 = sprintf('%s/sequences/%s/velodyne/%06d.bin', baseDir, seq, scanId1);
    labFile1 = sprintf('%s/sequences/%s/labels/%06d.label',   baseDir, seq, scanId1);
    binFile2 = sprintf('%s/sequences/%s/velodyne/%06d.bin', baseDir, seq, scanId2);
    labFile2 = sprintf('%s/sequences/%s/labels/%06d.label',   baseDir, seq, scanId2);

    % Leggi le due scansioni
    [pts1, sem1, inst1] = readSemanticKittiScan(binFile1, labFile1);
    [pts2, sem2, inst2] = readSemanticKittiScan(binFile2, labFile2);
    pts1_h = [pts1, ones(size(pts1,1),1)];      % [N x 4]
    pts2_h = [pts2, ones(size(pts2,1),1)];
    pts1_world = (T_w_velo_1 * pts1_h')';       % [N x 4]
    pts2_world = (T_w_velo_2 * pts2_h')';
    pts1_world = pts1_world(:,1:3);
    pts2_world = pts2_world(:,1:3);

    visualizeSemanticKittiFrame(pts1, sem1, inst1, 'semantic');
    visualizeSemanticKittiFrame(pts1, sem1, inst1, 'instance');
    visualizeSemanticKittiFrame(pts2, sem2, inst2, 'semantic');
    visualizeSemanticKittiFrame(pts2, sem2, inst2, 'instance');

    seg_t1 = buildSegmentsFromScan(pts1_world, sem1, inst1, thingIds, 20);
    seg_t2 = buildSegmentsFromScan(pts2_world, sem2, inst2, thingIds, 20);

    params = struct();
    params.maxMatchDist      = 3.0;       % metri
    params.moveThresh        = 5;       % metri
    params.sensorOrigin      = [0 0 0];   % posizione sensore
    params.occlusionAngleDeg = 3.0;       % tolleranza angolare (deg)

    results = semanticChangeDetection(seg_t1, seg_t2, params);

    fprintf('\n=== RISULTATI CHANGE DETECTION ===\n');
    fprintf('Oggetti persistenti      : %d\n', numel(results.persistent));
    fprintf('  di cui spostati        : %d\n', numel(results.movedPersistent));
    fprintf('  di cui stabili         : %d\n', numel(results.stablePersistent));
    fprintf('Oggetti nuovi            : %d\n', numel(results.newObjects));
    fprintf('Oggetti scomparsi        : %d\n', numel(results.disappeared));
    fprintf('  di cui occlusi         : %d\n', numel(results.disappearedOccluded));
    fprintf('  di cui visibili (veri) : %d\n', numel(results.disappearedVisible));

    plotChangeResultsKitti(pts1, pts2, seg_t1, seg_t2, results);

end

% traj = computeInstanceTrajectory(baseDir, seq, instId, frameList, useWorld, T_w_cam0, T_cam0_velo);
% plotInstanceTrajectory(traj);

scanId1 = 0;   % primo istante
scanId2 = 10;  % secondo istante
f1 = scanId1 + 1;   % gli indici in poses partono da 1
f2 = scanId2 + 1;
T_w_velo_1 = T_w_cam0{f1} * T_cam0_velo;
T_w_velo_2 = T_w_cam0{f2} * T_cam0_velo;

binFile1 = sprintf('%s/sequences/%s/velodyne/%06d.bin', baseDir, seq, scanId1);
labFile1 = sprintf('%s/sequences/%s/labels/%06d.label',   baseDir, seq, scanId1);
binFile2 = sprintf('%s/sequences/%s/velodyne/%06d.bin', baseDir, seq, scanId2);
labFile2 = sprintf('%s/sequences/%s/labels/%06d.label',   baseDir, seq, scanId2);

% Leggi le due scansioni
[pts1, sem1, inst1] = readSemanticKittiScan(binFile1, labFile1);
[pts2, sem2, inst2] = readSemanticKittiScan(binFile2, labFile2);
pts1_h = [pts1, ones(size(pts1,1),1)];      % [N x 4]
pts2_h = [pts2, ones(size(pts2,1),1)];
pts1_world = (T_w_velo_1 * pts1_h')';       % [N x 4]
pts2_world = (T_w_velo_2 * pts2_h')';
pts1_world = pts1_world(:,1:3);
pts2_world = pts2_world(:,1:3);

visualizeSemanticKittiFrame(pts1, sem1, inst1, 'semantic');
visualizeSemanticKittiFrame(pts1, sem1, inst1, 'instance');
visualizeSemanticKittiFrame(pts2, sem2, inst2, 'semantic');
visualizeSemanticKittiFrame(pts2, sem2, inst2, 'instance');

seg_t1 = buildSegmentsFromScan(pts1_world, sem1, inst1, thingIds, 20);
seg_t2 = buildSegmentsFromScan(pts2_world, sem2, inst2, thingIds, 20);

params = struct();
params.maxMatchDist      = 3.0;       % metri
params.moveThresh        = 5;       % metri
params.sensorOrigin      = [0 0 0];   % posizione sensore
params.occlusionAngleDeg = 3.0;       % tolleranza angolare (deg)

results = semanticChangeDetection(seg_t1, seg_t2, params);

fprintf('\n=== RISULTATI CHANGE DETECTION ===\n');
fprintf('Oggetti persistenti      : %d\n', numel(results.persistent));
fprintf('  di cui spostati        : %d\n', numel(results.movedPersistent));
fprintf('  di cui stabili         : %d\n', numel(results.stablePersistent));
fprintf('Oggetti nuovi            : %d\n', numel(results.newObjects));
fprintf('Oggetti scomparsi        : %d\n', numel(results.disappeared));
fprintf('  di cui occlusi         : %d\n', numel(results.disappearedOccluded));
fprintf('  di cui visibili (veri) : %d\n', numel(results.disappearedVisible));

plotChangeResultsKitti(pts1, pts2, seg_t1, seg_t2, results);

function results = semanticChangeDetection(seg_t1, seg_t2, params)
% semanticChangeDetection
%   Confronto di due segmentazioni semantiche 3D (t1 e t2) per change detection
%
% INPUT:
%   seg_t1  : struct array di oggetti al tempo 1
%   seg_t2  : struct array di oggetti al tempo 2
%   params  : struct con parametri (alcuni default sotto)
%       .maxMatchDist       : distanza max (m) per associare oggetti (stesso class)
%       .moveThresh         : soglia (m) di spostamento per considerare un oggetto "modificato"
%       .sensorOrigin       : [1x3] posizione sensore/camera (per occlusione)
%       .occlusionAngleDeg  : soglia angolare (deg) per considerare due oggetti sulla stessa linea di vista
%
% OUTPUT:
%   results: struct con:
%       .matches             : [Nmatch x 2] indici (i_t1, j_t2) di oggetti persistenti
%       .persistent          : struct array di oggetti persistenti (con info t1 e t2)
%       .movedPersistent     : sottoinsieme dei persistenti con spostamento > moveThresh
%       .stablePersistent    : sottoinsieme dei persistenti con spostamento <= moveThresh
%       .disappeared         : oggetti presenti solo in t1
%       .disappearedOccluded : oggetti scomparsi ma probabilmente occlusi
%       .disappearedVisible  : oggetti scomparsi non spiegati da occlusione (vera "scomparsa")
%       .newObjects          : oggetti presenti solo in t2
%
% NOTE:
%   - Questa è un'implementazione generica: potresti voler sostituire il matching
%     greedy con un assignment un po' più sofisticato (Hungarian, matchpairs, ecc.).
%   - L'occlusione è stimata in modo euristico: verifica se lungo la linea di vista
%     verso l'oggetto "scomparso" esiste un altro oggetto in t2, più vicino,
%     all'interno di una certa tolleranza angolare.

% Parametri di default
if nargin < 3
    params = struct();
end

if ~isfield(params, 'maxMatchDist'),      params.maxMatchDist = 3.0; end    % metri
if ~isfield(params, 'moveThresh'),        params.moveThresh   = 0.5; end   % metri
if ~isfield(params, 'sensorOrigin'),      params.sensorOrigin = [0 0 0]; end
if ~isfield(params, 'occlusionAngleDeg'), params.occlusionAngleDeg = 3.0; end

occlusionAngleRad = deg2rad(params.occlusionAngleDeg);

% Estrai centroidi e classi
n1 = numel(seg_t1);
n2 = numel(seg_t2);

C1 = zeros(n1,3);
C2 = zeros(n2,3);
I1 = zeros(n1,1);
I2 = zeros(n2,1);
class1 = strings(n1,1);
class2 = strings(n2,1);

for i = 1:n1
    C1(i,:) = seg_t1(i).centroid;
    I1(i,1) = seg_t1(i).id;
    class1(i) = string(seg_t1(i).class);
end

for j = 1:n2
    C2(j,:) = seg_t2(j).centroid;
    I2(j,1) = seg_t2(j).id;
    class2(j) = string(seg_t2(j).class);
end

% Matching oggetti tra t1 e t2 (greedy per distanza, con vincolo di classe)
matches = [];   % righe: [i_t1, j_t2, distanza]
used_t2 = false(n2, 1);

for i = 1:n1
    sameClassIdx = find(class2 == class1(i)); % stessa classe semantica
    sameInstanceIdx = find(I2 == I1(i));
    
    if isempty(sameClassIdx)
        continue;
    end

    if isempty(sameInstanceIdx)
        continue;
    end

    if I1(i) == 0
        continue;
    end
    
    % Calcola distanza alle candidate di t2
    diffs = C2(sameClassIdx,:) - C1(i,:);
    dists = sqrt(sum(diffs.^2, 2));
    
    % Prendi la più vicina
    [dmin, idxMin] = min(dists);

    dmin = 0;
    idxMin = sameInstanceIdx(1);
    
    if dmin <= params.maxMatchDist
        j = sameClassIdx(idxMin);
        % Controlla che t2(j) non sia già stato assegnato
        if ~used_t2(j)
            matches(end+1,:) = [i, j, dmin]; %#ok<AGROW>
            used_t2(j) = true;
        end
    end
end

% Oggetti persistenti
nMatch = size(matches,1);
persistentObjs = struct([]);

for k = 1:nMatch
    i = matches(k,1);
    j = matches(k,2);
    
    persistentObjs(k).id_t1      = seg_t1(i).id;
    persistentObjs(k).id_t2      = seg_t2(j).id;
    persistentObjs(k).class      = seg_t1(i).class;
    persistentObjs(k).centroid_t1 = seg_t1(i).centroid;
    persistentObjs(k).centroid_t2 = seg_t2(j).centroid;
    persistentObjs(k).bbox_t1    = getFieldOr(seg_t1(i), 'bbox', []);
    persistentObjs(k).bbox_t2    = getFieldOr(seg_t2(j), 'bbox', []);
    
    % Spostamento georeferenziato (norma della differenza dei centroidi)
    delta = persistentObjs(k).centroid_t2 - persistentObjs(k).centroid_t1;
    persistentObjs(k).displacement = norm(delta);
end

if nMatch > 0

% Separa persistent in moved / stable
movedMask  = [persistentObjs.displacement] > params.moveThresh;
stableMask = ~movedMask;

movedPersistent   = persistentObjs(movedMask);
stablePersistent  = persistentObjs(stableMask);

% Oggetti scomparsi (presenti solo in t1)
matched_t1 = false(n1,1);
matched_t1(matches(:,1)) = true;

disappearedIdx = find(~matched_t1);
disappeared = seg_t1(disappearedIdx);

% Oggetti nuovi (presenti solo in t2)
matched_t2 = false(n2,1);
matched_t2(matches(:,2)) = true;

newIdx = find(~matched_t2);
newObjects = seg_t2(newIdx);

% Verifica occlusione per gli oggetti scomparsi
disappearedOccluded = [];
disappearedVisible  = [];

for d = 1:numel(disappeared)
    obj = disappeared(d);
    c   = obj.centroid;

    % Vettore di vista dal sensore all'oggetto (tempo 1)
    v1 = c - params.sensorOrigin;
    dist1 = norm(v1);
    if dist1 == 0
        % Caso degenerato, lo consideriamo "visibile non occluso"
        if isempty(disappearedVisible)
            disappearedVisible = obj;
        else
            disappearedVisible(end+1) = obj;
        end
        continue;
    end
    v1 = v1 / dist1;  % direzione normalizzata

    % Verifica se qualche oggetto in t2 si trova lungo la stessa direzione, più vicino al sensore
    % (occlusione)
    isOccluded = false;
    for j = 1:n2
        cj = C2(j,:); % centroid t2
        v2 = cj - params.sensorOrigin;
        dist2 = norm(v2);
        if dist2 == 0
            continue;
        end
        v2 = v2 / dist2;
        
        % Angolo tra v1 e v2
        cosang = dot(v1, v2);
        cosang = max(min(cosang, 1), -1);
        ang = acos(cosang);
        
        % Se l'oggetto t2 è lungo la stessa linea di vista (angolo piccolo) e più vicino al sensore,
        % lo consideriamo occludente
        if (ang < occlusionAngleRad) && (dist2 < dist1)
            isOccluded = true;
            break;
        end
    end
    
    if isOccluded
        if isempty(disappearedOccluded)
            disappearedOccluded = obj;
        else
            disappearedOccluded(end+1) = obj;
        end
    else
        if isempty(disappearedVisible)
            disappearedVisible = obj;
        else
            disappearedVisible(end+1) = obj;
        end
    end
end

% Costruisci struct di output
results = struct();
results.matches             = matches;               % [i_t1, j_t2, dist]
results.persistent          = persistentObjs;
results.movedPersistent     = movedPersistent;
results.stablePersistent    = stablePersistent;
results.disappeared         = disappeared;
results.disappearedOccluded = disappearedOccluded;
results.disappearedVisible  = disappearedVisible;
results.newObjects          = newObjects;

else
    results = struct();
end


end

% Utility: leggi field se esiste, altrimenti valore di default
function val = getFieldOr(s, fieldName, defaultVal)
if isfield(s, fieldName)
    val = s.(fieldName);
else
    val = defaultVal;
end
end

function [points, semLabel, instLabel] = readSemanticKittiScan(binFile, labelFile)
% Legge una scansione da SemanticKITTI:
%   binFile  : .../velodyne/000000.bin
%   labelFile: .../labels/000000.label
%
% OUTPUT:
%   points   : [N x 3] (x,y,z)
%   semLabel : [N x 1] semantic id (uint16)
%   instLabel: [N x 1] instance id (uint16)

    % --- Point cloud (.bin) ---
    fid = fopen(binFile,'rb');
    if fid < 0
        error('Impossibile aprire il file bin: %s', binFile);
    end
    data = fread(fid, [4 inf], 'single')';   % [N x 4] (x,y,z,intensity)
    fclose(fid);
    points = double(data(:,1:3));           % [N x 3]

    % --- Labels (.label) ---
    fid = fopen(labelFile,'rb');
    if fid < 0
        error('Impossibile aprire il file label: %s', labelFile);
    end
    labels = fread(fid, 'uint32');
    fclose(fid);

    semLabel = uint16(bitand(labels, 65535, 'uint32'));  % 16 bit bassi
    instLabel = uint16(bitshift(labels, -16));           % 16 bit alti
end

function seg = buildSegmentsFromScan(points, semLabel, instLabel, thingIds, minPoints)
% buildSegmentsFromScan
%   Crea una lista di "oggetti" 3D a partire da una scansione SemanticKITTI raggruppando per
%   (semantic_id, instance_id).
%
% OUTPUT:
%   seg: struct array con campi:
%       .id        : instance id
%       .class     : semantic id
%       .centroid  : [1 x 3]
%       .bbox      : [1 x 6]
%       .indices   : indici dei punti in "points"

    if nargin < 5
        minPoints = 20;
    end

    [maskThing,~] = ismember(double(semLabel), thingIds);
    idxall = find(maskThing);
    points    = points(maskThing, :);
    semLabel  = double(semLabel(maskThing));
    instLabel = double(instLabel(maskThing));

    seg = struct('id',{},'class',{},'centroid',{},'bbox',{},'indices',{});

    if isempty(points)
        return;
    end

    pairs = [semLabel(:), instLabel(:)];
    uniquePairs = unique(pairs, 'rows');

    c = 0;
    for k = 1:size(uniquePairs,1)
        sem  = uniquePairs(k,1);
        inst = uniquePairs(k,2);

        mask = (semLabel == sem) & (instLabel == inst);
        pts  = points(mask, :);
        idx  = find(mask);          

        if size(pts,1) < minPoints
            continue;
        end

        c = c + 1;
        seg(c).id       = inst;
        seg(c).class    = sem;
        seg(c).centroid = mean(pts, 1);

        xmin = min(pts(:,1)); xmax = max(pts(:,1));
        ymin = min(pts(:,2)); ymax = max(pts(:,2));
        zmin = min(pts(:,3)); zmax = max(pts(:,3));
        seg(c).bbox     = [xmin xmax ymin ymax zmin zmax];
        seg(c).indices  = idxall(idx); 
    end
end

function plotChangeResultsKitti(pts1, pts2, seg_t1, seg_t2, results)
% plotChangeResultsKitti
%   Visualizza il risultato della change detection a livello di oggetto:
%   - frame t1: persistenti vs scomparsi (occlusi / veri)
%   - frame t2: persistenti stabili/spostati vs nuovi
%
% INPUT:
%   pts1, pts2 : [N x 3] punti (x,y,z) per t1 e t2 (già nel sistema che vuoi, locale o globale)
%   seg_t1     : segmenti/oggetti a t1 (buildSegmentsFromScan)
%   seg_t2     : segmenti/oggetti a t2
%   results    : output di semanticChangeDetection

% --- FIGURA 1: frame t1 -------------------------------------------------
figure('Name','Change detection - frame t1');
hold on; grid on; axis equal;
title('Frame t1: oggetti persistenti e scomparsi');

% Nuvola completa in grigio
hh = scatter3(pts1(:,1), pts1(:,2), pts1(:,3), 1, [0.8 0.8 0.8], '.');

% Persistenti (posizione a t1) - coloriamo tutti i punti dell'oggetto
kk = 0; hsp = [];
for k = 1:numel(results.persistent)
    instId = results.persistent(k).id_t1;
    cls    = results.persistent(k).class;
    segObj = findSegment(seg_t1, instId, cls);
    if ~isempty(segObj)
        kk = kk + 1;
        p = pts1(segObj.indices,:);
        hsp(kk) = scatter3(p(:,1), p(:,2), p(:,3), 5, 'b', '.'); % blu
    end
end

% Scomparsi ma occlusi
kk = 0; hso = [];
for k = 1:numel(results.disappearedOccluded)
    segObj = results.disappearedOccluded(k); % è già un seg_t1
    if isfield(segObj, 'indices') && ~isempty(segObj.indices)
        kk = kk + 1;
        p = pts1(segObj.indices,:);
        hso(kk) = scatter3(p(:,1), p(:,2), p(:,3), 8, 'y', 'filled'); % giallo
    else
        kk = kk + 1;
        c = segObj.centroid;
        hso(kk) = scatter3(c(1), c(2), c(3), 40, 'y', 'filled', 'd');
    end
end

% Scomparsi "veri" (non occlusi)
kk = 0; hsd = [];
for k = 1:numel(results.disappearedVisible)
    segObj = results.disappearedVisible(k); % seg_t1
    if isfield(segObj, 'indices') && ~isempty(segObj.indices)
        kk = kk + 1;
        p = pts1(segObj.indices,:);
        hsd(kk) = scatter3(p(:,1), p(:,2), p(:,3), 8, 'r', 'filled'); % rosso
    else
        kk = kk + 1;
        c = segObj.centroid;
        hsd(kk) = scatter3(c(1), c(2), c(3), 40, 'r', 'filled', 's');
    end
end

xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');

handles = [];
labels  = {};

if exist('hh','var') && ~isempty(hh)
    handles(end+1) = hh;
    labels{end+1}  = 'Punti t1';
end
if exist('hsp','var') && ~isempty(hsp)
    handles(end+1) = hsp(1);
    labels{end+1}  = 'Persistenti';
end
if exist('hso','var') && ~isempty(hso)
    handles(end+1) = hso(1);
    labels{end+1}  = 'Scomparsi occlusi';
end
if exist('hsd','var') && ~isempty(hsd)
    handles(end+1) = hsd(1);
    labels{end+1}  = 'Scomparsi "veri"';
end
if ~isempty(handles)
    legend(handles, labels, 'Location', 'bestoutside');
end

view(2);  % vista dall'alto, cambia con view(3) se ti piace di più
hold off;

% --- FIGURA 2: frame t2 -------------------------------------------------
figure('Name','Change detection - frame t2');
hold on; grid on; axis equal;
title('Frame t2: nuovi oggetti e persistenti');

% Nuvola completa in grigio
hh2 = scatter3(pts2(:,1), pts2(:,2), pts2(:,3), 1, [0.8 0.8 0.8], '.');

% Persistenti stabili (poco spostati)
kk = 0; hss = [];
for k = 1:numel(results.stablePersistent)
    instId = results.stablePersistent(k).id_t2;
    cls    = results.stablePersistent(k).class;
    segObj = findSegment(seg_t2, instId, cls);
    if ~isempty(segObj)
        kk = kk + 1;
        p = pts2(segObj.indices,:);
        hss(kk) = scatter3(p(:,1), p(:,2), p(:,3), 8, 'b', '.'); % blu
    end
end

% Persistenti spostati
kk = 0; hsm = [];
for k = 1:numel(results.movedPersistent)
    instId = results.movedPersistent(k).id_t2;
    cls    = results.movedPersistent(k).class;
    segObj = findSegment(seg_t2, instId, cls);
    if ~isempty(segObj)
        kk = kk + 1;
        p = pts2(segObj.indices,:);
        hsm(kk) = scatter3(p(:,1), p(:,2), p(:,3), 8, 'm', '.'); % magenta
    end
end

% Nuovi oggetti (solo in t2)
kk = 0; hsn = [];
for k = 1:numel(results.newObjects)
    segObj = results.newObjects(k); % seg_t2
    if isfield(segObj,'indices') && ~isempty(segObj.indices)
        kk = kk + 1;
        p = pts2(segObj.indices,:);
        hsn(kk) = scatter3(p(:,1), p(:,2), p(:,3), 8, 'g', '.'); % verde
    else
        kk = kk + 1;
        c = segObj.centroid;
        hsn(kk) = scatter3(c(1), c(2), c(3), 40, 'g', 'filled', 's');
    end
end

xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');

handles = [];
labels  = {};

if exist('hh2','var') && ~isempty(hh2)
    handles(end+1) = hh2;
    labels{end+1}  = 'Punti t2';
end
if exist('hss','var') && ~isempty(hss)
    handles(end+1) = hss(1);
    labels{end+1}  = 'Persistenti stabili';
end
if exist('hsm','var') && ~isempty(hsm)
    handles(end+1) = hsm(1);
    labels{end+1}  = 'Persistenti spostati';
end
if exist('hsn','var') && ~isempty(hsn)
    handles(end+1) = hsn(1);
    labels{end+1}  = 'Nuovi oggetti';
end
if ~isempty(handles)
    legend(handles, labels, 'Location', 'bestoutside');
end

view(2);
hold off;

end

% Helper per ritrovare il segmento dato (id, class)
function segObj = findSegment(segArray, instId, classVal)
    if isempty(segArray)
        segObj = [];
        return;
    end
    ids   = [segArray.id];
    class = [segArray.class];
    idx = find(ids == instId & class == classVal, 1, 'first');
    if isempty(idx)
        segObj = [];
    else
        segObj = segArray(idx);
    end
end

function T_w_cam0 = readKittiOdometryPoses(poseFile)
% Ritorna cell array T_w_cam0{frame} 4x4

    data = dlmread(poseFile);   % [N x 12]
    n = size(data,1);
    T_w_cam0 = cell(n,1);

    for i = 1:n
        M = reshape(data(i,:), [4,3])';   % 3x4 (row-major -> col-major)
        T = eye(4);
        T(1:3,1:4) = M;
        T_w_cam0{i} = T;
    end
end

function T_cam0_velo = readKittiCalibTr(calibFile)
% Estrae la matrice Tr (cam0 <- velo) come 4x4

    fid = fopen(calibFile, 'r');
    if fid < 0
        error('Impossibile aprire calib file: %s', calibFile);
    end

    T_cam0_velo = eye(4);
    while ~feof(fid)
        line = fgetl(fid);
        if startsWith(line, 'Tr:')
            nums = sscanf(line(4:end), '%f');
            M = reshape(nums, [4,3])';    % 3x4
            T_cam0_velo(1:3,1:4) = M;
            break;
        end
    end
    fclose(fid);
end

function instanceInfo = analyzeInstancePresence(baseDir, seq, frameIds, class_num, class_name, thingIds)
% analyzeInstancePresence
%   Analizza una sequenza SemanticKITTI e per ogni instance id (di classi "thing")
%   restituisce in quali frame appare, se ha buchi temporali, ecc.
%
% INPUT:
%   baseDir : root del dataset SemanticKITTI
%   seq     : stringa sequenza, es. '00'
%   frameIds: vettore di indici frame da analizzare (es. 0:100)
%   thingIds: vettore di semantic id "thing"
%
% OUTPUT:
%   instanceInfo: struct array con campi
%       .id        : instance id
%       .frames    : vettore di frame in cui appare
%       .firstFrame: primo frame
%       .lastFrame : ultimo frame
%       .numFrames : numero di frame in cui appare
%       .hasGaps   : true se ci sono buchi (riapparizioni non contigue)

    if nargin < 4
        error('Servono baseDir, seq, frameIds, thingIds');
    end

    map = containers.Map('KeyType','double', 'ValueType','any');
    map2 = containers.Map('KeyType','double', 'ValueType','any');

    for f = frameIds
        labFile = sprintf('%s/sequences/%s/labels/%06d.label', baseDir, seq, f);

        [semLabel, instLabel] = readSemanticKittiLabels(labFile);

        maskThing = ismember(double(semLabel), thingIds);
        inst = double(instLabel(maskThing));
        sem = semLabel(maskThing);

        inst = inst(inst > 0);    % rimuovi 0 / background
        if isempty(inst)
            continue;
        end

        [uInst,ia,ic] = unique(inst);
        uSem = sem(ia);
        for k = 1:numel(uInst)
            id = uInst(k);
            class = uSem(k);
            if map.isKey(id)
                map(id) = [map(id), f];
            else
                map(id) = f;
                map2(id) = class;
            end
        end
    end

    ids = cell2mat(map.keys);
    % sems = double(cell2mat(map2.values));
    instanceInfo = struct('id',{},'class',{},'frames',{},'firstFrame',{},'lastFrame',{},'numFrames',{},'hasGaps',{});

    for k = 1:numel(ids)
        id = ids(k);
        frames = sort(map(id));
        info.id         = id;
        class = find(class_num == map2(id));
        info.class      = class_name{class};
        info.frames     = frames;
        info.firstFrame = frames(1);
        info.lastFrame  = frames(end);
        info.numFrames  = numel(frames);
        info.hasGaps    = any(diff(frames) > 2);  % true se sparisce e poi riappare

        instanceInfo(k) = info;
    end
end

function [semLabel, instLabel] = readSemanticKittiLabels(labelFile)
% Legge solo le etichette SemanticKITTI (senza i punti)
    fid = fopen(labelFile,'rb');
    if fid < 0
        error('Impossibile aprire il file label: %s', labelFile);
    end
    labels = fread(fid, 'uint32');
    fclose(fid);

    semLabel  = uint16(bitand(labels, 65535, 'uint32'));
    instLabel = uint16(bitshift(labels, -16));
end

function traj = computeInstanceTrajectory(baseDir, seq, instanceId, frameList, useWorld, T_w_cam0, T_cam0_velo)
% computeInstanceTrajectory
%   Estrae la traiettoria (centroidi + info) di una singola instance id
%   lungo i frame specificati.
%
% INPUT:
%   baseDir   : root SemanticKITTI
%   seq       : stringa sequenza, es. '00'
%   instanceId: instance id da tracciare
%   frameList : vettore di frame in cui l'istanza appare (es. instanceInfo(k).frames)
%   useWorld  : true se vuoi trasformare in world coordinates (KITTI odometry)
%   T_w_cam0  : cell array di pose 4x4 (come da readKittiOdometryPoses), necessario se useWorld = true
%   T_cam0_velo : 4x4 trasformazione cam0 <- velo (da calib.txt), necessario se useWorld = true
%
% OUTPUT:
%   traj: struct con campi
%       .id          : instance id
%       .frames      : frameList
%       .centroids   : [M x 3] centroidi (solo dove esiste)
%       .semantics   : [M x 1] semantic id (primo punto dell'istanza)
%       .numPoints   : [M x 1] numero di punti dell'istanza
%       .exists      : [M x 1] logico, true se l'istanza è presente in quel frame
%       .refCentroid : [1 x 3] centroide nel primo frame in cui esiste
%       .displacement: [M x 1] distanza dal refCentroid (NaN dove non esiste)

    M = numel(frameList);
    centroids   = nan(M,3);
    semantics   = nan(M,1);
    numPoints   = zeros(M,1);
    exists      = false(M,1);

    for i = 1:M
        f = frameList(i);

        % --- file paths ---
        binFile = sprintf('%s/sequences/%s/velodyne/%06d.bin', baseDir, seq, f);
        labFile = sprintf('%s/sequences/%s/labels/%06d.label',   baseDir, seq, f);

        % --- leggi punti + label ---
        [pts, semLabel, instLabel] = readSemanticKittiScan(binFile, labFile);

        % --- trasformazione in world frame (opzionale) ---
        if useWorld
            frameIdx = f + 1; % poses indicizzate da 1
            T_w_velo = T_w_cam0{frameIdx} * T_cam0_velo;
            pts_h = [pts, ones(size(pts,1),1)];
            pts_w = (T_w_velo * pts_h')';
            pts = pts_w(:,1:3);
        end

        % --- seleziona punti con instanceId ---
        mask = (double(instLabel) == double(instanceId));
        if ~any(mask)
            % in teoria non dovrebbe succedere se frameList viene da analyzeInstancePresence,
            % ma per sicurezza mettiamo exists(i)=false
            continue;
        end

        objPts = pts(mask,:);
        objSem = double(semLabel(mask));

        exists(i)    = true;
        numPoints(i) = size(objPts,1);
        centroids(i,:) = mean(objPts,1);
        semantics(i) = mode(objSem);  % semantic id dominante dell'istanza
    end

    % centroide di riferimento = primo frame dove exists = true
    idxRef = find(exists, 1, 'first');
    if isempty(idxRef)
        refCentroid = [NaN NaN NaN];
        displacement = nan(M,1);
    else
        refCentroid = centroids(idxRef,:);
        diffs = centroids - refCentroid;
        displacement = sqrt(sum(diffs.^2, 2));
        displacement(~exists) = NaN;
    end

    traj = struct();
    traj.id          = instanceId;
    traj.frames      = frameList(:);
    traj.centroids   = centroids;
    traj.semantics   = semantics;
    traj.numPoints   = numPoints;
    traj.exists      = exists;
    traj.refCentroid = refCentroid;
    traj.displacement = displacement;
end

function plotInstanceTrajectory(traj)
% plotInstanceTrajectory
%   Visualizza:
%   1) Traiettoria 3D (o 2D XY) dei centroidi
%   2) Spostamento dal frame di riferimento in funzione del frame

    frames = traj.frames;
    C      = traj.centroids;
    dispR  = traj.displacement;

    % Usa solo punti dove esiste
    mask = traj.exists;
    frames_valid = frames(mask);
    C_valid      = C(mask,:);

    if isempty(frames_valid)
        warning('Nessun frame valido per instance id %d', traj.id);
        return;
    end

    % --- Figura 1: traiettoria spaziale ---
    figure('Name', sprintf('Instance %d - Trajectory', traj.id));
    subplot(1,2,1);
    hold on; grid on; axis equal;
    title(sprintf('Instance %d - spatial trajectory', traj.id));
    % linea dei centroidi
    plot3(C_valid(:,1), C_valid(:,2), C_valid(:,3), '-o');
    xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
    view(2); % dall'alto; cambia in view(3) se vuoi 3D interattivo
    % opzionale: numeri dei frame vicino ai punti
    for k = 1:numel(frames_valid)
        text(C_valid(k,1), C_valid(k,2), C_valid(k,3), sprintf('%d', frames_valid(k)), ...
            'FontSize', 6, 'Color', [0 0 0]);
    end
    hold off;

    % --- Figura 1, subplot 2: spostamento vs frame ---
    subplot(1,2,2);
    plot(frames, dispR, '-o');
    grid on;
    xlabel('Frame');
    ylabel('Displacement from first frame [m]');
    title('Displacement wrt reference frame');
end
