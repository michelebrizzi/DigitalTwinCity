baseDir = "D:\OneDrive - Universita degli Studi Roma Tre\Datasets\KITTI\odometry\dataset";
seq      = '00';
frameIds = 0:200;  % primi 200 frame, ad esempio

classes_num = [0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 52,...
    60, 70, 71, 72, 80, 81, 99, 252, 253, 254, 255, 256, 257, 258. 259];
class_name = {"unlabeled","outlier","car","bicycle","bus","motorcycle","on-rails","truck","other-vehicle",...
"person","bicyclist","motorcyclist","road","parking","sidewalk","other-ground","building","fence",...
"other-structure","lane-marking","vegetation","trunk","terrain","pole","traffic-sign","other-object",...
"moving-car","moving-bicyclist","moving-person","moving-motorcyclist","moving-on-rails","moving-bus",...
"moving-truck","moving-other-vehicle"};

thingIds = [ ...
    10 11 13 15 16 18 20 ...  % vehicles / bikes
    30 31 32 % persone e rider
];

instanceInfo = analyzeInstancePresence(baseDir, seq, frameIds, classes_num, class_name, thingIds);

% Instance che spariscono e poi riappaiono (hasGaps = true)
idxGappy = find([instanceInfo.hasGaps]);
gappyInstances = instanceInfo(idxGappy);

fprintf('Numero di oggetti che ricompaiono dopo un po'': %d\n', numel(gappyInstances));

% Ad esempio, prendi il primo e guarda in quali frame appare:
gappyInstances(1).id
gappyInstances(1).frames

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
        info.hasGaps    = any(diff(frames) > 1);  % true se sparisce e poi riappare

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
