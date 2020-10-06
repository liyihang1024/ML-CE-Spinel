#!/bin/bash

#########################################################
# Purpose:根据文件名新建文件夹并将文件移动到相应文件夹内#
# Author: Yihang LI                         			#
# Updated Date: Sep. 25, 2019                           #
# Usage: 将此脚本放在需批处理的文件同路径下，运行脚本.  #
#        ./mkdir&mv.sh                  				#
#########################################################

for i in *.xlsx
do mkdir $(basename ${i} .xlsx)
	mv $i ./$(basename ${i} .xlsx)
done
