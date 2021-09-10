#!/bin/bash


./build/change_detection/all_scenes_comparison /home/edith/liebnas_mnt/PlaneReconstructions/GH30_kitchen -r /home/edith/liebnas_mnt/PlaneReconstructions/Results/GH30_kitchen -c /home/edith/Projects/TidyUpVisionPipeline/v4r_ppf/cfg/ppf_pose_estimation_config.ini
./build/change_detection/all_scenes_comparison /home/edith/liebnas_mnt/PlaneReconstructions/GH30_office  -r /home/edith/liebnas_mnt/PlaneReconstructions/Results/GH30_office  -c /home/edith/Projects/TidyUpVisionPipeline/v4r_ppf/cfg/ppf_pose_estimation_config.ini
./build/change_detection/all_scenes_comparison /home/edith/liebnas_mnt/PlaneReconstructions/GH30_living  -r /home/edith/liebnas_mnt/PlaneReconstructions/Results/GH30_living  -c /home/edith/Projects/TidyUpVisionPipeline/v4r_ppf/cfg/ppf_pose_estimation_config.ini
./build/change_detection/all_scenes_comparison /home/edith/liebnas_mnt/PlaneReconstructions/Arena        -r /home/edith/liebnas_mnt/PlaneReconstructions/Results/Arena        -c /home/edith/Projects/TidyUpVisionPipeline/v4r_ppf/cfg/ppf_pose_estimation_config.ini
./build/change_detection/all_scenes_comparison /home/edith/liebnas_mnt/PlaneReconstructions/KennyLab     -r /home/edith/liebnas_mnt/PlaneReconstructions/Results/KennyLab     -c /home/edith/Projects/TidyUpVisionPipeline/v4r_ppf/cfg/ppf_pose_estimation_config.ini
