function cam=getCameraParams(fname,platform)
% returns camera parameters and file information
% fname: video or image(s) file name(s) i.e. mymovie.mov or IMG_3797.jpg
% platform: camera name i.e. 'iPhone 6s'
if iscell(fname); fname=fname{1}; end
if nargin==1; platform='iPhone 6s'; end

isImageFile = strcmpi(extension(fname),'jpg'); %true for images
switch platform
    case 'iPhone 6s'
        %pixelSize = 0.0011905; %(mm) on a side, 12um
        sensorSizeMM = [4.80 3.60]; %(mm) CMOS sensor
        FocalLengthMM = 4.15; %(mm) iPhone 6s from EXIF
        %focalLengthPixels = FocalLengthMM / sensorSizeMM(1) * cam.Width
        %FOV = [atand(cam.Width/2/focalLengthPixels) atand(cam.Height/2/focalLengthPixels)]*2; % (deg) camera field of view
        FOV = atand(sensorSizeMM/2/FocalLengthMM)*2; % (deg) camea field of view

        if isImageFile  % 12MP image 4032x3024
            cam = imfinfo(fname); cam.imageType='image'; cam.videoFlag=false; cam.kltBlockSize=[21 21];
            [cam.PathName, cam.FileName] = pfsplit(cam.Filename);
            
            Skew = 0;
            FocalLength = [3486 3486];
            PrincipalPoint = [cam.Width cam.Height]/2 + 0.5;
            IntrinsicMatrix = [FocalLength(1) 0 0; Skew FocalLength(2) 0; PrincipalPoint 1];
            radialDistortion = [0 0 0];
            

%             focalLength = cam.DigitalCamera.FocalLength / sensorSize(1) * cam.Width;
%             FOV = [atand(cam.Width/2/focalLength) atand(cam.Height/2/focalLength)]*2; % (deg) camera field of view
        else  % 4k video 3840x2160
            cam.V = VideoReader(fname); cam.imageType='video'; cam.videoFlag=true; cam.kltBlockSize=[51 51];
            cam.PathName=cam.V.Path;  cam.FileName=cam.V.Name;
            
            cam.Width=cam.V.Width; cam.Height=cam.V.Height; 
            if cam.Width>cam.Height; cam.Orientation=1; else; cam.Orientation=6; end % 1 = landscape, 6 = vertical
            diagonalRatio = norm([4032 3024])./norm([3840 2160]); %ratio of image to video frame diagonal lengths:  https://photo.stackexchange.com/questions/86075/does-the-iphones-focal-length-differ-when-taking-video-vs-photos

            Skew = 0;
            FocalLength = [3486 3486]*diagonalRatio;
            PrincipalPoint = [cam.Width cam.Height]/2 + 0.5;
            IntrinsicMatrix = [FocalLength(1) 0 0; Skew FocalLength(2) 0; PrincipalPoint 1];
            radialDistortion = [0 0 0];
        end
    case 'iPhone X'
end

switch cam.Orientation
    case 1
        cam.OrientationComment = 'Horizontal';
    case 6
        cam.OrientationComment = 'Vertical';
end

%Define camera parameters
cam.params = cameraParameters('IntrinsicMatrix', IntrinsicMatrix, 'RadialDistortion', radialDistortion);
%load('/Users/glennjocher/Downloads/IMG_4136 Stills/cameraParamsVideo.mat');  cam.params=cameraParams;
%load('/Users/glennjocher/Downloads/IMG_4137 Burst/cameraParams.mat'); cam.params=cameraParams;

%Pre-define interpolation grid for use with imwarp or interp2
[X,Y]=meshgrid(single(1:cam.Width),single(1:cam.Height));  a=ones(numel(X),3,'single');  a(:,1)=X(:);  a(:,2)=Y(:);  cam.Ixy=a;
end
