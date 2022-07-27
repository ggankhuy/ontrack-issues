echo ------------------
setpci -s 2f:00.0 CAP_EXP+30.L=00000001
lspci -s 2f:00.0 -vv | grep Lnk
lspci -s 2f:00.0 -xxx | grep 80:
echo ------------------
setpci -s 2f:00.0 CAP_EXP+30.L=00000002
lspci -s 2f:00.0 -vv | grep Lnk
lspci -s 2f:00.0 -xxx | grep 80:
echo ------------------
setpci -s 2f:00.0 CAP_EXP+30.L=00000004
lspci -s 2f:00.0 -vv | grep Lnk
lspci -s 2f:00.0 -xxx | grep 80:
