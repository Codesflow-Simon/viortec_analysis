% Measured variables
syms a1 a2 d
% Known variables
r=0.3;

theta = a2-a1;

% We have the formula
% d^2 = r^2 + r^2 - 2*r*r*cos(phi)

phi = acos((2*r^2-d^2)/(2*r^2));
gamma = sqrt(phi^2 - (theta-180)^2);

% Calculate derivatives of gamma w.r.t theta and d
dgamma_dtheta = diff(gamma, a2);
dgamma_dd = diff(gamma,d);

% Display the results
disp('dgamma_dtheta = ');
disp(simplify(dgamma_dtheta));
disp('dgamma_dd = ');
disp(simplify(dgamma_dd));

% Evaluate these expression on a grid of different theta and d values
theta_vals = linspace(0,180,100) + 0.0001;
d_vals = linspace(0,0.6,100) + 0.0001;
[THETA,D] = meshgrid(theta_vals,d_vals);
dgamma_dtheta_vals = subs(dgamma_dtheta, {a1,a2,d}, {0,THETA,D});
dgamma_dd_vals = subs(dgamma_dd, {a1,a2,d}, {0,THETA,D});

% Plot the results as a heatmap
figure(1);
clf;
subplot(2,1,1);
imagesc(dgamma_dtheta_vals);
axis image;
colorbar();
xlabel('theta');
ylabel('d');
title('dgamma_dtheta');

subplot(2,1,2);
imagesc(dgamma_dd_vals);
axis image;
colorbar();
xlabel('theta');
ylabel('d');
title('dgamma_dd');
