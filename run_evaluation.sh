#!/bin/bash
#| sort -n | tail -5 | head -1 --> runs evaluation for the 5th last folder
last_Arena_folder=$(printf "%s\n" /home/edith/liebnas_mnt/PlaneReconstructions/Results/Arena/*  | tail -1)
last_KennyLab_folder=$(printf "%s\n" /home/edith/liebnas_mnt/PlaneReconstructions/Results/KennyLab/*  | tail -1)
last_GH30_living_folder=$(printf "%s\n" /home/edith/liebnas_mnt/PlaneReconstructions/Results/GH30_living/*  | tail -1)
last_GH30_office_folder=$(printf "%s\n" /home/edith/liebnas_mnt/PlaneReconstructions/Results/GH30_office/*  | tail -1)
last_GH30_kitchen_folder=$(printf "%s\n" /home/edith/liebnas_mnt/PlaneReconstructions/Results/GH30_kitchen/*  | tail -1)

./build/evaluation/evaluation $last_GH30_kitchen_folder	/home/edith/liebnas_mnt/PlaneReconstructions/Annotations/GH30_kitchen
./build/evaluation/evaluation $last_GH30_office_folder	/home/edith/liebnas_mnt/PlaneReconstructions/Annotations/GH30_office
./build/evaluation/evaluation $last_GH30_living_folder	/home/edith/liebnas_mnt/PlaneReconstructions/Annotations/GH30_living
./build/evaluation/evaluation $last_Arena_folder	/home/edith/liebnas_mnt/PlaneReconstructions/Annotations/Arena
./build/evaluation/evaluation $last_KennyLab_folder	/home/edith/liebnas_mnt/PlaneReconstructions/Annotations/KennyLab

