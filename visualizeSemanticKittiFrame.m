function visualizeSemanticKittiFrame(points, semLabel, instLabel, mode)
% visualizeSemanticKittiFrame
%   Visualizza una point cloud KITTI/SemanticKITTI colorata:
%   - per semantic class (mode = 'semantic')
%   - per instance id   (mode = 'instance')
%
% INPUT:
%   points   : [N x 3] (x,y,z) nel sistema di coordinate che vuoi (locale o world)
%   semLabel : [N x 1] uint16 semantic id
%   instLabel: [N x 1] uint16 instance id
%   mode     : 'semantic' (default) oppure 'instance'

    if nargin < 4
        mode = 'semantic';
    end

    classes_num = [0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 52,...
        60, 70, 71, 72, 80, 81, 99, 252, 253, 254, 255, 256, 257, 258, 259];
    
    class_name = {"unlabeled","outlier","car","bicycle","bus","motorcycle","on-rails","truck","other-vehicle",...
        "person","bicyclist","motorcyclist","road","parking","sidewalk","other-ground","building","fence",...
        "other-structure","lane-marking","vegetation","trunk","terrain","pole","traffic-sign","other-object",...
        "moving-car","moving-bicyclist","moving-person","moving-motorcyclist","moving-on-rails","moving-bus",...
        "moving-truck","moving-other-vehicle"};

    points   = double(points);
    semLabel = double(semLabel);
    instLabel = double(instLabel);

    switch lower(mode)
        case 'semantic'
            [labs,~,ic] = unique(semLabel);
            C = colorsFromSemanticKitti(semLabel);
            ttl = 'SemanticKITTI - colori per classe semantica';

            leg = [];
            figure('Name', ttl);
            hold on
            for i = 1:length(labs)
                qq = (ic==i);

                cl = class_name{classes_num == labs(i)};
                leg = [leg cl];

                scatter3(points(qq,1), points(qq,2), points(qq,3), 1, C(qq,:), '.');
            end
            grid on;
            xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
            legend(leg)
            title(ttl);
            view(2); % view(2): vista dall'alto; view(3): vista 3D

        case 'instance'
            [labs,~,ic] = unique(instLabel);
            C = colorsFromInstanceId(instLabel);
            ttl = 'SemanticKITTI - colori per instance id';


            figure('Name', ttl);
            hold on
            for i = 1:length(labs)
                qq = (ic==i);
                scatter3(points(qq,1), points(qq,2), points(qq,3), 1, C(qq,:), '.');
            end
            grid on;
            xlabel('X [m]'); ylabel('Y [m]'); zlabel('Z [m]');
            title(ttl);
            view(2); % view(2): vista dall'alto; view(3): vista 3D
        otherwise
            error('mode deve essere ''semantic'' oppure ''instance''');
    end
end

function C = colorsFromSemanticKitti(semLabel)
% colorsFromSemanticKitti
%   Assegna un colore RGB ad ogni punto in base al semantic id di SemanticKITTI.
%
% INPUT:
%   semLabel : [N x 1] vettore di semantic id
% OUTPUT:
%   C        : [N x 3] colori in [0,1]

    semLabel = double(semLabel(:));
    N = numel(semLabel);
    C = zeros(N,3); % default nero

    % Mappa (id -> colore). Puoi modificarla come vuoi.
    % (valori indicativi, ma già abbastanza leggibili)
    mapId   = [ ...
         0   % unlabeled
         1   % outlier
        10   % car
        11   % bicycle
        13   % bus
        15   % motorcycle
        16   % on-rails
        18   % truck
        20   % other-vehicle
        30   % person
        31   % bicyclist
        32   % motorcyclist
        40   % road
        44   % parking
        48   % sidewalk
        49   % other-ground
        50   % building
        51   % fence
        52   % other-structure
        60   % lane-marking
        70   % vegetation
        71   % trunk
        72   % terrain
        80   % pole
        81   % traffic-sign
        99   % other-object
        252  % moving-car
        253  % moving-bicyclist
        254  % moving-person
        255  % moving-motorcyclist
    ];

    mapCol = [ ...
        0.0 0.0 0.0;      % 0 - unlabeled (nero)
        0.3 0.3 0.3;      % 1 - outlier (grigio)
        0.0 0.0 0.9;      % 10 - car (blu)
        0.0 0.7 0.7;      % 11 - bicycle (ciano)
        0.0 0.5 0.0;      % 13 - bus (verde scuro)
        0.6 0.0 0.6;      % 15 - motorcycle (viola)
        0.3 0.0 0.6;      % 16 - on-rails
        0.0 0.4 0.0;      % 18 - truck
        0.0 0.8 0.5;      % 20 - other-vehicle
        1.0 0.0 0.0;      % 30 - person (rosso)
        1.0 0.4 0.0;      % 31 - bicyclist (arancio)
        1.0 0.0 0.5;      % 32 - motorcyclist (fucsia)
        0.5 0.5 0.5;      % 40 - road (grigio medio)
        0.6 0.6 0.2;      % 44 - parking
        0.9 0.9 0.9;      % 48 - sidewalk (quasi bianco)
        0.4 0.3 0.2;      % 49 - other-ground
        0.7 0.7 0.7;      % 50 - building
        0.3 0.3 0.7;      % 51 - fence
        0.5 0.2 0.7;      % 52 - other-structure
        1.0 1.0 0.0;      % 60 - lane-marking (giallo)
        0.0 0.7 0.0;      % 70 - vegetation (verde)
        0.0 0.4 0.2;      % 71 - trunk
        0.4 0.6 0.0;      % 72 - terrain
        0.0 0.0 0.0;      % 80 - pole (nero)
        1.0 1.0 1.0;      % 81 - traffic-sign (bianco)
        0.7 0.0 0.0;      % 99 - other-object
        0.0 0.0 1.0;      % 252 - moving-car (blu intenso)
        0.0 1.0 1.0;      % 253 - moving-bicyclist
        1.0 0.0 1.0;      % 254 - moving-person
        0.5 0.0 0.5;      % 255 - moving-motorcyclist
    ];

    % Per ogni punto, assegna il colore
    for i = 1:numel(mapId)
        id = mapId(i);
        mask = (semLabel == id);
        C(mask, :) = repmat(mapCol(i,:), sum(mask), 1);
    end

    % I semantic id non elencati rimangono nero (0,0,0)
end

function C = colorsFromInstanceId(instLabel)
% colorsFromInstanceId
%   Assegna un colore deterministico ad ogni instance id (stesso id -> stesso colore),
%   usando una mappa tipo HSV.
%
% INPUT:
%   instLabel : [N x 1] uint16 instance id
% OUTPUT:
%   C         : [N x 3] colori RGB in [0,1]

    instLabel = double(instLabel(:));
    N = numel(instLabel);
    C = zeros(N,3);

    uniqueIds = unique(instLabel);
    % zero (0) di solito background / niente istanza → lasciamo grigio
    uniqueIds(uniqueIds == 0) = [];

    for k = 1:numel(uniqueIds)
        id = uniqueIds(k);
        mask = (instLabel == id);

        % Colore deterministico basato sull'id:
        % golden ratio trick per variare l'hue
        hue = mod(id * 0.61803398875, 1.0);
        sat = 0.8;
        val = 0.95;
        rgb = hsv2rgb([hue sat val]);

        C(mask, :) = repmat(rgb, sum(mask), 1);
    end

    % Background (id = 0) → grigio chiaro
    mask0 = (instLabel == 0);
    C(mask0,:) = repmat([0.6 0.6 0.6], sum(mask0), 1);
end
