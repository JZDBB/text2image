#!/bin/zsh

for file in /home/yn/Desktop/text2image/models/*
do
if [ -f "$file" ]
then 
  echo "$file is file"
fi
done

awk 'BEGIN{
for(i=1; i<=5; i++) 
print (i*i+1)
}'