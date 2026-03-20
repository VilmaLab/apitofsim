#!/bin/bash

# this program extract data from Gaussian and ORCA output for ACDC sim.


for i in $*
do
  file=$i #input = file.log or file.out

  ####
  filebase=`echo $file | rev | cut -c5- | rev`
  if [ ! -e $filebase.out ]
  then
    echo "ERROR: File $filebase.out does not exist."
    continue
  fi
  if [ ! -e $filebase.log ]
  then
    echo "ERROR: File $filebase.log does not exist."
    continue
  fi

  ####
  echo "Processing file $filebase.log and $filebase.out"
  newfile=$(basename $filebase)

  ####
  ####
  echo " - generating ${newfile}_en.dat"
  ENdlpno=`grep "FINAL SINGLE POINT ENERGY" $filebase.out | awk '{print $5}'`
  #ENtmdcorr=`grep "Thermal correction to Gibbs Free Energy" $filebase.log | awk '{print $7}'`
  ## G16: ENzpe=`grep "Zero-point correction" $filebase.log | awk '{print $3}'`
  ENzpe=`grep "Zero point energy" $filebase.log | awk '{print $5}'`
  EN=`echo $ENdlpno+$ENzpe | bc -l`
  echo $EN > ${newfile}_en.dat
  ####
  echo " - generating ${newfile}_rot.dat"
  ## G16: grep "Rotational temperatures" $filebase.log | awk '{print $6,$5,$4}'  | xargs -n 1 > ${newfile}_rot.dat
  m=1.43877688 #magic number or so-called Bulgarian constant [cm-1 to K]
  grep "Rotational constants in cm-1" $filebase.log | tail -n 1 | awk -v m=$m '{print $7*m,$6*m,$5*m}' | xargs -n 1 > ${newfile}_rot.dat
  ####
  echo " - generating ${newfile}_vib.dat"
  ## G16: grep "Frequencies" $filebase.log | awk '{print $3,$4,$5}' | xargs -n 1 | awk '{print $1*1.4388}' > ${newfile}_vib.dat
  grep "E(vib)   ..." $filebase.log | awk -v m=$m '{print $2*m}' > ${newfile}_vib.dat
  ####
  echo " Done :-D"
done
