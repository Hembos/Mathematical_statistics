format long g
pkg load interval


data1 = csvread("./data/Канал 1_700nm_0.03.csv")
data2 = csvread("./data/Канал 2_700nm_0.03.csv")

data1(1,:) = []
data2(1,:) = []

x1 = data1(:,1)
x2 = data2(:,1)
eps = 1e-4

n = transpose(1:length(x1))
X = [n.^0 n]

[data1_tau, w1] = L_1_minimization(X, x1 - eps, x1 + eps);
[data2_tau, w2] = L_1_minimization(X, x2 - eps, x2 + eps);

fileID1 = fopen('data/Ch1.txt','w');
fileID2 = fopen('data/Ch2.txt','w');
fprintf(fileID1,'%g %g\n', data1_tau(1), data1_tau(2));
fprintf(fileID2,'%g %g\n', data2_tau(1), data2_tau(2));
for c = 1 : length(w1)
  fprintf(fileID1, "%g\n", w1(c));
  fprintf(fileID2, "%g\n", w2(c));
end
fclose(fileID1);
fclose(fileID2);
