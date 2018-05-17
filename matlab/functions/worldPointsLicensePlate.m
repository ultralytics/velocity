function worldPoints=worldPointsLicensePlate()
%Returns x, y coordinates of license plate outline in meters
%https://en.wikipedia.org/wiki/Vehicle_registration_plate
pSize = [.3725 .1275]; %[0.36 0.13]; %(m) license plate size (Chile)
worldPoints =  [ 1    -1
                 1     1
                -1     1
                -1    -1].*(pSize/2);

