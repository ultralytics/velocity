function R = cam2ned()
%x_ned = R*x_cam;

% +X_ned (NORTH) = +Z_cam
% +Y_ned (EAST)  = +X_cam
% +Z_ned (DOWN)  = +Y_cam

R = [0 0 1
     1 0 0
     0 1 0]; 

