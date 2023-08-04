A=`setpci -s 2f:00.0 00.L`

echo A1: $A
A=$(( 16#$A ))
echo A2: $A
B=$((A & 16#FFFF0000))
echo B1: $B
B=`printf '%x\n' $B`
echo B2: $B

# CLEAR THE BIT  

echo ----
echo clear the bit...

A=`setpci -s 2f:00.0 04.L`

echo A1: $A
A=$(( 16#$A ))
echo A2: $A
C=$(( ~ 0x01 ))
echo C: $C
B=$((A & 16#$C))
echo B1: $B
B=`printf '%x\n' $B`
echo B2: $B

# SET THE BIT

echo ----
echo set the bit...

A=`setpci -s 2f:00.0 04.L`

echo A1: $A
A=$(( 16#$A ))
echo A2: $A
B=$((A | 16#FFFF))
echo B1: $B
B=`printf '%x\n' $B`
echo B2: $B
