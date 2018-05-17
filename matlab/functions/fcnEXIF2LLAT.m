function llat = fcnEXIF2LLAT(E)
%E = image exif info i.e. E = imfinfo('beach.jpg')
%llat = [lat, long, alt, time]
G=E.GPSInfo;
C=E.DigitalCamera;
day = datenum([C.DateTimeOriginal '.' C.SubsecTimeOriginal],'yyyy:mm:dd HH:MM:SS.FFF'); %fractional day since 00/00/000

llat(1) = dms2degrees(G.GPSLatitude)*hemisphere2sign(G.GPSLatitudeRef);
llat(2) = dms2degrees(G.GPSLongitude)*hemisphere2sign(G.GPSLongitudeRef);
llat(3) = G.GPSAltitude;
llat(4) = (day-floor(day))*86400; %seconds since midnight
%llat(4) = day*86400; % seconds since January 1st, year 0 (0000:01:01)


function y=hemisphere2sign(x)
% converts hemisphere strings 'N', 'S', 'E', 'W' to signs 1, -1, 1, -1
y=zeros(size(x));
y(x=='N' | x=='E') = 1;
y(x=='S' | x=='W') = -1;

